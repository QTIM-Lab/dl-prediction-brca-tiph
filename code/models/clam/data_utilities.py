# Imports
import os
import pandas as pd
import h5py
import numpy as np  # For random selection in augmentation
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit


# Class: TCGABRCA_MIL_Dataset
class TCGABRCA_MIL_Dataset(Dataset):
    def __init__(self, base_data_path='TCGA-BRCA', experimental_strategy='All', label=None, features_h5_dir=None,
                 train_size=0.70, val_size=0.15, test_size=0.15, n_folds=10, seed=42, transform=None):
        assert experimental_strategy in ('All', 'DiagnosticSlide', 'TissueSlide')
        assert features_h5_dir is not None
        assert label in (
            'hallmark_angiogenesis',
            'hallmark_epithelial_mesenchymal_transition',
            'hallmark_fatty_acid_metabolism',
            'hallmark_oxidative_phosphorylation',
            'hallmark_glycolysis',
            'kegg_antigen_processing_and_presentation',
            'gobp_t_cell_mediated_cytotoxicity',
            'gobp_b_cell_proliferation',
            'kegg_cell_cycle',
            'immunosuppression'
        )
        assert train_size + val_size + test_size == 1
        assert n_folds > 0
        assert seed is not None

        # Class variables
        self.base_data_path = base_data_path
        self.experimental_strategy = experimental_strategy
        self.features_h5_dir = features_h5_dir
        self.label = label
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.n_folds = n_folds
        self.seed = seed
        self.curr_fold = 0
        self.curr_split = 'train'

        # Choose experimental strategy
        if experimental_strategy in ('DiagnosticSlide', 'TissueSlide'):
            if experimental_strategy == 'DiagnosticSlide':
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'DiagnosticSlide.csv'))
                assert len(data) == 1133, f'The Diagnostic Slide subset size should be 1133, not {len(data)}.'
            else:
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'TissueSlide.csv'))
                assert len(data) == 1978, f'The Tissue Slide subset size should be 1978, not {len(data)}.'
            # Get folders and SVS files
            folders, svs_files = data['folder_name'].values, data['svs_name']
            svs_fpaths = []
            for folder, svs_file in zip(folders, svs_files):
                svs_fpath = os.path.join(base_data_path, 'WSI', folder, svs_file)
                svs_fpaths.append(svs_fpath)
        else:
            svs_fpaths = self.get_all_folders_and_svs_fpaths()

        # Build dictionary of image paths by Case ID
        self.svs_fpaths = svs_fpaths
        self.svs_fpaths_dict = self.get_svs_fpaths_dict()

        # Get the .h5 files with features (original.h5)
        self.features = self.get_features_h5()

        # Load ssGSEA Scores
        self.ssgsea_scores_dict = self.load_tcga_brca_ssgsea_scores()
        ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict = dict(), dict()
        for idx, label_ in enumerate(self.ssgsea_scores_dict['label_names']):
            ssgsea_scores_idx_label_dict[idx] = label_
            ssgsea_scores_label_idx_dict[label_] = idx
        self.ssgsea_scores_label_idx_dict, self.ssgsea_scores_idx_label_dict = ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict

        # Build dataset dictionary
        self.dataset_dict, self.wsi_genex_label_dict, self.features_h5_dict = self.build_dataset_dicts()

        # Split into train, validation, and test sets by Case ID
        self.train_dict, self.val_dict, self.test_dict = self.split_dataset()

        # Set label threshold
        self.label_threshold = 0

        self.transform = transform

        return

    # Method: Get all folders and SVS paths
    def get_all_folders_and_svs_fpaths(self):
        wsi_directory = os.path.join(self.base_data_path, 'WSI')
        folders = [f for f in os.listdir(wsi_directory) if not f.startswith('.')]
        svs_fpaths = []
        for folder in folders:
            folder_contents = [c for c in os.listdir(os.path.join(wsi_directory, folder)) if not c.startswith('.')]
            svs_files = [s for s in folder_contents if s.endswith('svs')]
            for svs in svs_files:
                svs_fpath = os.path.join(wsi_directory, folder, svs)
                svs_fpaths.append(svs_fpath)
        return svs_fpaths

    # Method: Create a dictionary of SVS paths by Case ID
    def get_svs_fpaths_dict(self):
        svs_fpaths_dict = {}
        for path in self.svs_fpaths:
            cid = self.get_case_id(wsi_path_or_name=path)
            if cid not in svs_fpaths_dict:
                svs_fpaths_dict[cid] = []
            svs_fpaths_dict[cid].append(path)
        return svs_fpaths_dict

    # Method: Get the list of .h5 files with features
    def get_features_h5(self):
        features_h5_files = []
        for f_dir in self.features_h5_dir:
            f_dir_folders = [f for f in os.listdir(f_dir) if os.path.isdir(os.path.join(f_dir, f))]
            for folder in f_dir_folders:
                folder_files = [f for f in os.listdir(os.path.join(f_dir, folder)) if not f.startswith('.')]
                if 'original.h5' in folder_files:
                    features_h5_files.append(os.path.join(f_dir, folder, 'original.h5'))
        return features_h5_files

    # Method: Load TCGA_BRCA ssGSEA Scores
    def load_tcga_brca_ssgsea_scores(self):
        df = pd.read_csv(os.path.join(self.base_data_path, 'Annotations', 'TCGA_BRCA_ssGSEA_Scores.csv'))
        sgsea_scores_dict = {}
        for col_name in df.columns:
            if col_name == 'Unnamed: 0':
                sgsea_scores_dict['label_names'] = df['Unnamed: 0'].values
            else:
                sgsea_scores_dict[col_name] = df[col_name].values
        return sgsea_scores_dict

    # Method: Get Case ID from path or name
    def get_case_id(self, wsi_path_or_name, mode='simple'):
        assert mode in ('simple', 'extended')
        parsed_path = wsi_path_or_name.split('/')[-1]
        parsed_path = parsed_path.split('.')[0]
        if mode == 'simple':
            parsed_path = parsed_path.split('-')[0:3]
        else:
            parsed_path = parsed_path.split('-')[0:4]
        case_id = '-'.join(parsed_path)
        return case_id

    # Method: Build the dataset dictionary
    def build_dataset_dicts(self):
        dataset_dict = {
            'case_id': [],
            'svs_fpath': [],
            'features_h5': [],
            'ssgea_id': [],
            'ssgsea_scores': []
        }
        # Process scores and create a mapping of Case ID to entries
        wsi_genex_label_dict = {}
        for w in self.ssgsea_scores_dict.keys():
            if w != 'label_names':
                case_id = self.get_case_id(wsi_path_or_name=w)
                if case_id not in wsi_genex_label_dict:
                    wsi_genex_label_dict[case_id] = []
                wsi_genex_label_dict[case_id].append(w)
        # Map feature file names by Case ID
        features_h5_dict = {}
        for f in self.features:
            case_id = self.get_case_id(wsi_path_or_name=f.split('/')[-2])
            if case_id not in features_h5_dict:
                features_h5_dict[case_id] = []
            features_h5_dict[case_id].append(f)
        # Process WSIs
        for case_id in self.svs_fpaths_dict.keys():
            if case_id in wsi_genex_label_dict and case_id in features_h5_dict:
                for svs_path in self.svs_fpaths_dict[case_id]:
                    wsi_fname = os.path.splitext(svs_path.split('/')[-1])[0]
                    feature_h5_fname = ''
                    for fname in features_h5_dict[case_id]:
                        if wsi_fname in fname.split('/'):
                            feature_h5_fname = fname
                    ssgea_scores_list = wsi_genex_label_dict[case_id]
                    case_id_ext = self.get_case_id(wsi_path_or_name=svs_path, mode='extended')
                    for ssgea in ssgea_scores_list:
                        ssgea_ext = self.get_case_id(wsi_path_or_name=ssgea, mode='extended')
                        if case_id_ext == ssgea_ext and feature_h5_fname in features_h5_dict[case_id]:
                            dataset_dict['case_id'].append(case_id)
                            dataset_dict['svs_fpath'].append(svs_path)
                            dataset_dict['features_h5'].append(feature_h5_fname)
                            dataset_dict['ssgea_id'].append(ssgea)
                            dataset_dict['ssgsea_scores'].append(self.ssgsea_scores_dict[ssgea])
        assert len(dataset_dict['case_id']) == len(dataset_dict['svs_fpath'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['features_h5'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgea_id'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgsea_scores'])
        return dataset_dict, wsi_genex_label_dict, features_h5_dict

    # Method: Split the dataset into train, validation, and test sets
    def split_dataset(self):
        trainval_dict, train_dict, val_dict, test_dict = {}, {}, {}, {}
        gss_trainval_test = GroupShuffleSplit(
            n_splits=self.n_folds,
            train_size=self.train_size + self.val_size,
            random_state=self.seed
        )
        gss_train_val = GroupShuffleSplit(
            n_splits=1,
            train_size=(self.train_size / (self.train_size + self.val_size)),
            random_state=self.seed
        )
        groups = self.dataset_dict['case_id']
        X = self.dataset_dict['features_h5']
        y = self.dataset_dict['ssgsea_scores']
        for fold, (train_index, test_index) in enumerate(gss_trainval_test.split(X, y, groups)):
            trainval_dict[fold] = {
                'case_id': [self.dataset_dict['case_id'][i] for i in train_index],
                'svs_fpath': [self.dataset_dict['svs_fpath'][i] for i in train_index],
                'features_h5': [self.dataset_dict['features_h5'][i] for i in train_index],
                'ssgea_id': [self.dataset_dict['ssgea_id'][i] for i in train_index],
                'ssgsea_scores': [self.dataset_dict['ssgsea_scores'][i] for i in train_index],
            }
            test_dict[fold] = {
                'case_id': [self.dataset_dict['case_id'][i] for i in test_index],
                'svs_fpath': [self.dataset_dict['svs_fpath'][i] for i in test_index],
                'features_h5': [self.dataset_dict['features_h5'][i] for i in test_index],
                'ssgea_id': [self.dataset_dict['ssgea_id'][i] for i in test_index],
                'ssgsea_scores': [self.dataset_dict['ssgsea_scores'][i] for i in test_index],
            }
        for fold in range(self.n_folds):
            groups = trainval_dict[fold]['case_id']
            X = trainval_dict[fold]['features_h5']
            y = trainval_dict[fold]['ssgsea_scores']
            for _, (train_index, test_index) in enumerate(gss_train_val.split(X, y, groups)):
                train_dict[fold] = {
                    'case_id': [trainval_dict[fold]['case_id'][i] for i in train_index],
                    'svs_fpath': [trainval_dict[fold]['svs_fpath'][i] for i in train_index],
                    'features_h5': [trainval_dict[fold]['features_h5'][i] for i in train_index],
                    'ssgea_id': [trainval_dict[fold]['ssgea_id'][i] for i in train_index],
                    'ssgsea_scores': [trainval_dict[fold]['ssgsea_scores'][i] for i in train_index]
                }
                val_dict[fold] = {
                    'case_id': [trainval_dict[fold]['case_id'][i] for i in test_index],
                    'svs_fpath': [trainval_dict[fold]['svs_fpath'][i] for i in test_index],
                    'features_h5': [trainval_dict[fold]['features_h5'][i] for i in test_index],
                    'ssgea_id': [trainval_dict[fold]['ssgea_id'][i] for i in test_index],
                    'ssgsea_scores': [trainval_dict[fold]['ssgsea_scores'][i] for i in test_index]
                }
        for fold in range(self.n_folds):
            assert len(self.dataset_dict['case_id']) == len(train_dict[fold]['case_id']) + len(val_dict[fold]['case_id']) + len(test_dict[fold]['case_id'])
        return train_dict, val_dict, test_dict

    # Method: Change fold
    def select_fold(self, fold):
        assert fold in range(self.n_folds)
        self.curr_fold = fold
        return

    # Method: Select split
    def select_split(self, split):
        assert split in ('train', 'validation', 'test')
        self.curr_split = split
        return

    def __len__(self):
        if self.curr_split == 'train':
            return len(self.train_dict[self.curr_fold]['case_id'])
        elif self.curr_split == 'validation':
            return len(self.val_dict[self.curr_fold]['case_id'])
        else:
            return len(self.test_dict[self.curr_fold]['case_id'])

    # Method: __getitem__ with slide-level and tile-level augmentation
    def __getitem__(self, idx):
        # Select the dataset dictionary based on current split
        if self.curr_split == 'train':
            dataset_dict = self.train_dict[self.curr_fold]
        elif self.curr_split == 'validation':
            dataset_dict = self.val_dict[self.curr_fold]
        else:
            dataset_dict = self.test_dict[self.curr_fold]

        # Get Case ID, SVS path, and feature file path
        case_id = dataset_dict['case_id'][idx]
        svs_path = dataset_dict['svs_fpath'][idx]
        features_h5 = dataset_dict['features_h5'][idx]
        with h5py.File(features_h5, "r") as f:
            features = f["features"][()]

        # --- Slide-level augmentation decision ---
        # With 20% probability, augment the slide; otherwise, use original features
        if np.random.rand() < 0.2:
            # --- Tile-level augmentation ---
            # For 50% of the tiles, randomly select one option among 5 choices:
            # the original and the 4 augmentation files.
            folder = os.path.dirname(features_h5)
            options = ["original.h5", "hed_rs0.h5", "hed_rs1.h5", "hed_rs2.h5", "hed_rs3.h5"]
            n_tiles = features.shape[0]
            tile_mask = np.random.rand(n_tiles) < 0.5  # 50% chance per tile (only in augmented slides)
            if tile_mask.sum() > 0:
                indices = np.where(tile_mask)[0]
                # For each selected tile, choose one option uniformly at random
                chosen_options = np.random.choice(options, size=len(indices))
                for opt in np.unique(chosen_options):
                    opt_indices = indices[chosen_options == opt]
                    if opt == "original.h5":
                        # If the chosen option is original, do nothing (tile remains as is)
                        continue
                    else:
                        aug_path = os.path.join(folder, opt)
                        with h5py.File(aug_path, "r") as f_aug:
                            features_aug = f_aug["features"][()]
                        features[opt_indices] = features_aug[opt_indices]
        # -----------------------------------------
        features = torch.from_numpy(features)

        # Get ssGSEA score and binarize based on threshold
        ssgea_id = dataset_dict['ssgea_id'][idx]
        if self.label:
            ssgsea_scores = dataset_dict['ssgsea_scores'][idx][self.ssgsea_scores_label_idx_dict[self.label]]
            ssgsea_scores = float(ssgsea_scores)
            ssgsea_scores = 1 if ssgsea_scores > self.label_threshold else 0
        else:
            ssgsea_scores = -1

        c_subtype = dataset_dict['c_subtype'][idx]
        c_subtype_label = dataset_dict['c_subtype_label'][idx]

        input_data_dict = {
            'case_id': case_id,
            'svs_path': svs_path,
            'features_h5': features_h5,
            'features': features,
            'ssgea_id': ssgea_id,
            'ssgsea_scores': ssgsea_scores,
            'c_subtype': c_subtype,
            'c_subtype_label': c_subtype_label
        }
        return input_data_dict


# Class: TCGABRCA_MIL_DatasetRegression
class TCGABRCA_MIL_DatasetRegression(TCGABRCA_MIL_Dataset):
    def __init__(self, base_data_path='TCGA-BRCA', experimental_strategy='All', label=None, features_h5_dir=None,
                 train_size=0.7, val_size=0.15, test_size=0.15, n_folds=10, seed=42, transform=None):
        super().__init__(base_data_path, experimental_strategy, label, features_h5_dir,
                         train_size, val_size, test_size, n_folds, seed, transform)
        return

    def __getitem__(self, idx):
        if self.curr_split == 'train':
            dataset_dict = self.train_dict[self.curr_fold]
        elif self.curr_split == 'validation':
            dataset_dict = self.val_dict[self.curr_fold]
        else:
            dataset_dict = self.test_dict[self.curr_fold]

        case_id = dataset_dict['case_id'][idx]
        svs_path = dataset_dict['svs_fpath'][idx]
        features_h5 = dataset_dict['features_h5'][idx]
        with h5py.File(features_h5, "r") as f:
            features = f["features"][()]

        # --- Data augmentation (same slide-level and tile-level strategy) ---
        if np.random.rand() < 0.2:
            folder = os.path.dirname(features_h5)
            options = ["original.h5", "hed_rs0.h5", "hed_rs1.h5", "hed_rs2.h5", "hed_rs3.h5"]
            n_tiles = features.shape[0]
            tile_mask = np.random.rand(n_tiles) < 0.5
            if tile_mask.sum() > 0:
                indices = np.where(tile_mask)[0]
                chosen_options = np.random.choice(options, size=len(indices))
                for opt in np.unique(chosen_options):
                    opt_indices = indices[chosen_options == opt]
                    if opt == "original.h5":
                        continue
                    else:
                        aug_path = os.path.join(folder, opt)
                        with h5py.File(aug_path, "r") as f_aug:
                            features_aug = f_aug["features"][()]
                        features[opt_indices] = features_aug[opt_indices]
        # ----------------------------------------------------
        features = torch.from_numpy(features)
        ssgea_id = dataset_dict['ssgea_id'][idx]
        ssgsea_scores = dataset_dict['ssgsea_scores'][idx][self.ssgsea_scores_label_idx_dict[self.label]]
        ssgsea_scores = float(ssgsea_scores)
        ssgsea_scores_bin = 1 if ssgsea_scores > self.label_threshold else 0

        input_data_dict = {
            'case_id': case_id,
            'svs_path': svs_path,
            'features_h5': features_h5,
            'features': features,
            'ssgea_id': ssgea_id,
            'ssgsea_scores': ssgsea_scores,
            'ssgsea_scores_bin': ssgsea_scores_bin
        }
        return input_data_dict


# Class: TCGABRCA_MIL_DatasetClinicalSubtype
class TCGABRCA_MIL_DatasetClinicalSubtype(Dataset):
    def __init__(self, base_data_path='TCGA-BRCA', experimental_strategy='All', label=None, features_h5_dir=None,
                 train_size=0.70, val_size=0.15, test_size=0.15, n_folds=10, seed=42, transform=None):
        assert experimental_strategy in ('All', 'DiagnosticSlide', 'TissueSlide')
        assert features_h5_dir is not None
        if label:
            assert label in (
                'hallmark_angiogenesis',
                'hallmark_epithelial_mesenchymal_transition',
                'hallmark_fatty_acid_metabolism',
                'hallmark_oxidative_phosphorylation',
                'hallmark_glycolysis',
                'kegg_antigen_processing_and_presentation',
                'gobp_t_cell_mediated_cytotoxicity',
                'gobp_b_cell_proliferation',
                'kegg_cell_cycle',
                'immunosuppression'
            )
        assert train_size + val_size + test_size == 1
        assert n_folds > 0
        assert seed is not None

        # Class variables
        self.base_data_path = base_data_path
        self.experimental_strategy = experimental_strategy
        self.features_h5_dir = features_h5_dir
        self.label = label
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.n_folds = n_folds
        self.seed = seed
        self.curr_fold = 0
        self.curr_split = 'train'

        if experimental_strategy in ('DiagnosticSlide', 'TissueSlide'):
            if experimental_strategy == 'DiagnosticSlide':
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'DiagnosticSlide.csv'))
                assert len(data) == 1133, f'The Diagnostic Slide subset size should be 1133, not {len(data)}.'
            else:
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'TissueSlide.csv'))
                assert len(data) == 1978, f'The Tissue Slide subset size should be 1978, not {len(data)}.'
            folders, svs_files = data['folder_name'].values, data['svs_name']
            svs_fpaths = []
            for folder, svs_file in zip(folders, svs_files):
                svs_fpath = os.path.join(base_data_path, 'WSI', folder, svs_file)
                svs_fpaths.append(svs_fpath)
        else:
            svs_fpaths = self.get_all_folders_and_svs_fpaths()

        self.svs_fpaths = svs_fpaths
        self.svs_fpaths_dict = self.get_svs_fpaths_dict()
        self.features = self.get_features_h5()
        self.ssgsea_scores_dict = self.load_tcga_brca_ssgsea_scores()
        ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict = dict(), dict()
        for idx, label_ in enumerate(self.ssgsea_scores_dict['label_names']):
            ssgsea_scores_idx_label_dict[idx] = label_
            ssgsea_scores_label_idx_dict[label_] = idx
        self.ssgsea_scores_label_idx_dict, self.ssgsea_scores_idx_label_dict = ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict

        self.tcga_brca_labels_dict = self.load_tcga_brca_labels()
        tcga_brca_label_idx_dict = {
            "HER2+/HR+": 0,
            "HER2+/HR-": 1,
            "HER2-/HR+": 2,
            "HER2-/HR-": 3
        }
        tcga_brca_idx_label_dict = {}
        for k, v in tcga_brca_label_idx_dict.items():
            tcga_brca_idx_label_dict[v] = k
        self.tcga_brca_label_idx_dict, self.tcga_brca_idx_label_dict = tcga_brca_label_idx_dict, tcga_brca_idx_label_dict

        self.dataset_dict, self.wsi_genex_label_dict, self.features_h5_dict = self.build_dataset_dicts()
        self.train_dict, self.val_dict, self.test_dict = self.split_dataset()
        self.label_threshold = 0
        self.transform = transform

        return

    def get_all_folders_and_svs_fpaths(self):
        wsi_directory = os.path.join(self.base_data_path, 'WSI')
        folders = [f for f in os.listdir(wsi_directory) if not f.startswith('.')]
        svs_fpaths = []
        for folder in folders:
            folder_contents = [c for c in os.listdir(os.path.join(wsi_directory, folder)) if not c.startswith('.')]
            svs_files = [s for s in folder_contents if s.endswith('svs')]
            for svs_f in svs_files:
                svs_fpath = os.path.join(wsi_directory, folder, svs_f)
                svs_fpaths.append(svs_fpath)
        return svs_fpaths

    def get_svs_fpaths_dict(self):
        svs_fpaths_dict = {}
        for path in self.svs_fpaths:
            cid = self.get_case_id(wsi_path_or_name=path)
            if cid not in svs_fpaths_dict:
                svs_fpaths_dict[cid] = []
            svs_fpaths_dict[cid].append(path)
        return svs_fpaths_dict

    def get_features_h5(self):
        features_h5_files = []
        for f_dir in self.features_h5_dir:
            f_dir_folders = [f for f in os.listdir(f_dir) if os.path.isdir(os.path.join(f_dir, f))]
            for folder in f_dir_folders:
                folder_files = [f for f in os.listdir(os.path.join(f_dir, folder)) if not f.startswith('.')]
                if 'original.h5' in folder_files:
                    features_h5_files.append(os.path.join(f_dir, folder, 'original.h5'))
        return features_h5_files

    def load_tcga_brca_ssgsea_scores(self):
        df = pd.read_csv(os.path.join(self.base_data_path, 'Annotations', 'TCGA_BRCA_ssGSEA_Scores.csv'))
        sgsea_scores_dict = {}
        for col_name in df.columns:
            if col_name == 'Unnamed: 0':
                sgsea_scores_dict['label_names'] = df['Unnamed: 0'].values
            else:
                sgsea_scores_dict[col_name] = df[col_name].values
        return sgsea_scores_dict

    def load_tcga_brca_labels(self):
        tcga_brca_labels = pd.read_excel(os.path.join(self.base_data_path, 'Annotations', 'breast_cancer_tcga_labels_1.xlsx'))
        tcga_brca_labels = tcga_brca_labels.copy()[['CLID', 'er_status_by_ihc', 'pr_status_by_ihc', 'HER2.newly.derived']]
        tcga_brca_labels_dict = {}
        for _, r in tcga_brca_labels.iterrows():
            clid = self.get_case_id(r['CLID'])
            er = r['er_status_by_ihc']
            pr = r['pr_status_by_ihc']
            her2 = r['HER2.newly.derived']
            c_subtype, c_subtype_label = self.compute_clinical_subtype(er, pr, her2)
            if c_subtype is not None:
                if clid not in tcga_brca_labels_dict:
                    tcga_brca_labels_dict[clid] = {}
                tcga_brca_labels_dict[clid]["c_subtype"] = c_subtype
                tcga_brca_labels_dict[clid]["c_subtype_label"] = c_subtype_label
                tcga_brca_labels_dict[clid]["er"] = er
                tcga_brca_labels_dict[clid]["pr"] = pr
                tcga_brca_labels_dict[clid]["her2"] = her2
        return tcga_brca_labels_dict

    def get_case_id(self, wsi_path_or_name, mode='simple'):
        assert mode in ('simple', 'extended')
        parsed_path = wsi_path_or_name.split('/')[-1]
        parsed_path = parsed_path.split('.')[0]
        if mode == 'simple':
            parsed_path = parsed_path.split('-')[0:3]
        else:
            parsed_path = parsed_path.split('-')[0:4]
        case_id = '-'.join(parsed_path)
        return case_id

    def compute_clinical_subtype(self, er, pr, her2):
        # Compute hormone receptor (HR) status
        hr = None
        if er == "Positive" or pr == "Positive":
            hr = "Positive"
        elif er == "Negative" and pr == "Negative":
            hr = "Negative"
        c_subtype = None
        c_subtype_label = -1
        if her2 == "Positive" and hr == "Positive":
            c_subtype = "HER2+/HR+"
            c_subtype_label = 0
        elif her2 == "Positive" and hr == 'Negative':
            c_subtype = "HER2+/HR-"
            c_subtype_label = 1
        elif her2 == "Negative" and hr == "Positive":
            c_subtype = "HER2-/HR+"
            c_subtype_label = 2
        elif her2 == "Negative" and hr == "Negative":
            c_subtype = "HER2-/HR-"
            c_subtype_label = 3
        return c_subtype, c_subtype_label

    def build_dataset_dicts(self):
        dataset_dict = {
            'case_id': [],
            'svs_fpath': [],
            'features_h5': [],
            'ssgea_id': [],
            'ssgsea_scores': [],
            'c_subtype': [],
            'c_subtype_label': []
        }
        wsi_genex_label_dict = {}
        for w in self.ssgsea_scores_dict.keys():
            if w != 'label_names':
                case_id = self.get_case_id(wsi_path_or_name=w)
                if case_id not in wsi_genex_label_dict:
                    wsi_genex_label_dict[case_id] = []
                wsi_genex_label_dict[case_id].append(w)
        features_h5_dict = {}
        for f in self.features:
            case_id = self.get_case_id(wsi_path_or_name=f.split('/')[-2])
            if case_id not in features_h5_dict:
                features_h5_dict[case_id] = []
            features_h5_dict[case_id].append(f)
        for case_id in self.svs_fpaths_dict.keys():
            if case_id in wsi_genex_label_dict and case_id in features_h5_dict:
                for svs_path in self.svs_fpaths_dict[case_id]:
                    wsi_fname = os.path.splitext(svs_path.split('/')[-1])[0]
                    feature_h5_fname = ''
                    for fname in features_h5_dict[case_id]:
                        if wsi_fname in fname.split('/'):
                            feature_h5_fname = fname
                    ssgea_scores_list = wsi_genex_label_dict[case_id]
                    case_id_ext = self.get_case_id(wsi_path_or_name=svs_path, mode='extended')
                    for ssgea in ssgea_scores_list:
                        ssgea_ext = self.get_case_id(wsi_path_or_name=ssgea, mode='extended')
                        if case_id_ext == ssgea_ext and feature_h5_fname in features_h5_dict[case_id] and case_id in self.tcga_brca_labels_dict.keys():
                            dataset_dict['case_id'].append(case_id)
                            dataset_dict['svs_fpath'].append(svs_path)
                            dataset_dict['features_h5'].append(feature_h5_fname)
                            dataset_dict['ssgea_id'].append(ssgea)
                            dataset_dict['ssgsea_scores'].append(self.ssgsea_scores_dict[ssgea])
                            dataset_dict['c_subtype'].append(self.tcga_brca_labels_dict[case_id]['c_subtype'])
                            dataset_dict['c_subtype_label'].append(self.tcga_brca_labels_dict[case_id]['c_subtype_label'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['svs_fpath'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['features_h5'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgea_id'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgsea_scores'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['c_subtype'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['c_subtype_label'])
        return dataset_dict, wsi_genex_label_dict, features_h5_dict

    def split_dataset(self):
        trainval_dict, train_dict, val_dict, test_dict = {}, {}, {}, {}
        gss_trainval_test = GroupShuffleSplit(
            n_splits=self.n_folds,
            train_size=self.train_size + self.val_size,
            random_state=self.seed
        )
        gss_train_val = GroupShuffleSplit(
            n_splits=1,
            train_size=(self.train_size / (self.train_size + self.val_size)),
            random_state=self.seed
        )
        groups = self.dataset_dict['case_id']
        X = self.dataset_dict['features_h5']
        y = self.dataset_dict['ssgsea_scores']
        for fold, (train_index, test_index) in enumerate(gss_trainval_test.split(X, y, groups)):
            trainval_dict[fold] = {
                'case_id': [self.dataset_dict['case_id'][i] for i in train_index],
                'svs_fpath': [self.dataset_dict['svs_fpath'][i] for i in train_index],
                'features_h5': [self.dataset_dict['features_h5'][i] for i in train_index],
                'ssgea_id': [self.dataset_dict['ssgea_id'][i] for i in train_index],
                'ssgsea_scores': [self.dataset_dict['ssgsea_scores'][i] for i in train_index],
                'c_subtype': [self.dataset_dict['c_subtype'][i] for i in train_index],
                'c_subtype_label': [self.dataset_dict['c_subtype_label'][i] for i in train_index]
            }
            test_dict[fold] = {
                'case_id': [self.dataset_dict['case_id'][i] for i in test_index],
                'svs_fpath': [self.dataset_dict['svs_fpath'][i] for i in test_index],
                'features_h5': [self.dataset_dict['features_h5'][i] for i in test_index],
                'ssgea_id': [self.dataset_dict['ssgea_id'][i] for i in test_index],
                'ssgsea_scores': [self.dataset_dict['ssgsea_scores'][i] for i in test_index],
                'c_subtype': [self.dataset_dict['c_subtype'][i] for i in test_index],
                'c_subtype_label': [self.dataset_dict['c_subtype_label'][i] for i in test_index]
            }
        for fold in range(self.n_folds):
            groups = trainval_dict[fold]['case_id']
            X = trainval_dict[fold]['features_h5']
            y = trainval_dict[fold]['ssgsea_scores']
            for _, (train_index, test_index) in enumerate(gss_train_val.split(X, y, groups)):
                train_dict[fold] = {
                    'case_id': [trainval_dict[fold]['case_id'][i] for i in train_index],
                    'svs_fpath': [trainval_dict[fold]['svs_fpath'][i] for i in train_index],
                    'features_h5': [trainval_dict[fold]['features_h5'][i] for i in train_index],
                    'ssgea_id': [trainval_dict[fold]['ssgea_id'][i] for i in train_index],
                    'ssgsea_scores': [trainval_dict[fold]['ssgsea_scores'][i] for i in train_index],
                    'c_subtype': [trainval_dict[fold]['c_subtype'][i] for i in train_index],
                    'c_subtype_label': [trainval_dict[fold]['c_subtype_label'][i] for i in train_index]
                }
                val_dict[fold] = {
                    'case_id': [trainval_dict[fold]['case_id'][i] for i in test_index],
                    'svs_fpath': [trainval_dict[fold]['svs_fpath'][i] for i in test_index],
                    'features_h5': [trainval_dict[fold]['features_h5'][i] for i in test_index],
                    'ssgea_id': [trainval_dict[fold]['ssgea_id'][i] for i in test_index],
                    'ssgsea_scores': [trainval_dict[fold]['ssgsea_scores'][i] for i in test_index],
                    'c_subtype': [trainval_dict[fold]['c_subtype'][i] for i in test_index],
                    'c_subtype_label': [trainval_dict[fold]['c_subtype_label'][i] for i in test_index]
                }
        for fold in range(self.n_folds):
            assert len(self.dataset_dict['case_id']) == len(train_dict[fold]['case_id']) + len(val_dict[fold]['case_id']) + len(test_dict[fold]['case_id'])
        return train_dict, val_dict, test_dict

    def select_fold(self, fold):
        assert fold in range(self.n_folds)
        self.curr_fold = fold
        return

    def select_split(self, split):
        assert split in ('train', 'validation', 'test')
        self.curr_split = split
        return

    def __len__(self):
        if self.curr_split == 'train':
            return len(self.train_dict[self.curr_fold]['case_id'])
        elif self.curr_split == 'validation':
            return len(self.val_dict[self.curr_fold]['case_id'])
        else:
            return len(self.test_dict[self.curr_fold]['case_id'])

    # Method: __getitem__ with slide-level and tile-level augmentation
    def __getitem__(self, idx):
        if self.curr_split == 'train':
            dataset_dict = self.train_dict[self.curr_fold]
        elif self.curr_split == 'validation':
            dataset_dict = self.val_dict[self.curr_fold]
        else:
            dataset_dict = self.test_dict[self.curr_fold]

        case_id = dataset_dict['case_id'][idx]
        svs_path = dataset_dict['svs_fpath'][idx]
        features_h5 = dataset_dict['features_h5'][idx]
        with h5py.File(features_h5, "r") as f:
            features = f["features"][()]

        # --- Slide-level augmentation decision ---
        # With 20% probability, perform augmentation on this slide.
        if np.random.rand() < 0.2:
            # --- Tile-level augmentation ---
            # For 50% of the tiles, choose randomly one option among 5: original, hed_rs0.h5, hed_rs1.h5, hed_rs2.h5, hed_rs3.h5.
            folder = os.path.dirname(features_h5)
            options = ["original.h5", "hed_rs0.h5", "hed_rs1.h5", "hed_rs2.h5", "hed_rs3.h5"]
            n_tiles = features.shape[0]
            tile_mask = np.random.rand(n_tiles) < 0.5  # 50% chance per tile (only for augmented slides)
            if tile_mask.sum() > 0:
                indices = np.where(tile_mask)[0]
                chosen_options = np.random.choice(options, size=len(indices))
                for opt in np.unique(chosen_options):
                    opt_indices = indices[chosen_options == opt]
                    if opt == "original.h5":
                        # If original is chosen, leave the tile unchanged.
                        continue
                    else:
                        aug_path = os.path.join(folder, opt)
                        with h5py.File(aug_path, "r") as f_aug:
                            features_aug = f_aug["features"][()]
                        features[opt_indices] = features_aug[opt_indices]
        # -----------------------------------------
        features = torch.from_numpy(features)

        ssgea_id = dataset_dict['ssgea_id'][idx]
        if self.label:
            ssgsea_scores = dataset_dict['ssgsea_scores'][idx][self.ssgsea_scores_label_idx_dict[self.label]]
            ssgsea_scores = float(ssgsea_scores)
            ssgsea_scores = 1 if ssgsea_scores > self.label_threshold else 0
        else:
            ssgsea_scores = -1

        c_subtype = dataset_dict['c_subtype'][idx]
        c_subtype_label = dataset_dict['c_subtype_label'][idx]

        input_data_dict = {
            'case_id': case_id,
            'svs_path': svs_path,
            'features_h5': features_h5,
            'features': features,
            'ssgea_id': ssgea_id,
            'ssgsea_scores': ssgsea_scores,
            'c_subtype': c_subtype,
            'c_subtype_label': c_subtype_label
        }
        return input_data_dict