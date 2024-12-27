# Imports
import os
import pandas as pd
import h5py

# PyTorch Imports
import torch
from torch.utils.data import Dataset

# Sklearn Imports
from sklearn.model_selection import GroupShuffleSplit



# MGHPathDataset
class MGHPathDataset(Dataset):
    def __init__(self, base_data_path=None, features_h5_dir=None, seed=42, transform=None):
        
        assert base_data_path is not None
        assert features_h5_dir is not None
        assert seed is not None

        # Class variables
        self.base_data_path = base_data_path
        self.features_h5_dir = features_h5_dir
        self.seed = seed

        # Get the .pt files with feature
        self.features = self.get_features_h5()

        # Class variables
        self.transform = transform
        
        return


    # Method: Get the list of .pt files in the features directory
    def get_features_h5(self):

        features_h5_files = list()

        f_dir_folders = [f for f in os.listdir(self.features_h5_dir) if os.path.isdir(os.path.join(self.features_h5_dir, f))]            
        for folder in f_dir_folders:
            folder_files = [f for f in os.listdir(os.path.join(self.features_h5_dir, folder)) if not f.startswith('.')]
            if 'original.h5' in folder_files:
                features_h5_files.append(os.path.join(self.features_h5_dir, folder, 'original.h5'))
        # print(len(features_h5_files))

        return features_h5_files



# Class: OhioStatePathDataset
class OhioStatePathDataset(Dataset):
    def __init__(self, base_data_path=None, features_h5_dir=None, seed=42, transform=None):
        
        assert base_data_path is not None
        assert features_h5_dir is not None
        assert seed is not None

        # Class variables
        self.base_data_path = base_data_path
        self.features_h5_dir = features_h5_dir
        self.seed = seed

        # Get .TIFFs paths
        self.tiffs_fpaths = self.get_ohiostatewsi_folders_and_tiff_fpaths()

        # Get mapping of the .TIFF path to the renamed version
        self.id2fpath_wsi = self.get_id2fpath_wsi()

        # Get the .pt files with feature
        self.features = self.get_features_h5()

        # Class variables
        self.transform = transform
        
        return


    # Method: Get all folders and .TIFF paths from the Ohio State WSI dataset
    def get_ohiostatewsi_folders_and_tiff_fpaths(self):

        # Get the list of .TIFF files without subdirectory (i.e., root)
        r_tiffs_ = [t for t in os.listdir(self.base_data_path) if not t.startswith('.')]
        r_tiffs_ = [t for t in r_tiffs_ if not os.path.isdir(os.path.join(self.base_data_path, t))]
        r_tiffs_ = [t for t in r_tiffs_ if t.endswith('.tiff')]
        r_tiffs = [os.path.join(self.base_data_path, t) for t in r_tiffs_]

        # Now, go through the folders (BATCHX)
        r_subdirs = [sb for sb in os.listdir(self.base_data_path) if os.path.isdir(os.path.join(self.base_data_path, sb))]
        r_subdirs = [sb for sb in r_subdirs if not sb.startswith('.')]
        subdirs_fpaths = list()
        for subdir in r_subdirs:
            if subdir in ('BATCH2', 'BATCH3', 'BATCH4'):
                subdir_path = os.path.join(self.base_data_path, subdir)
                subdir_tiffs = [st for st in os.listdir(subdir_path) if not st.startswith('.')]
                subdir_tiffs = [st for st in subdir_tiffs if st.endswith('.tiff')]
                subdir_tiffs = [os.path.join(subdir_path, st) for st in subdir_tiffs]
                subdirs_fpaths.extend(subdir_tiffs)
        
        # Create final list of fpaths
        tiffs_fpaths = r_tiffs + subdirs_fpaths
        # print(len(tiffs_fpaths))
        # print(tiffs_fpaths)

        # The original dataset currently has 70 .TIFF files
        assert len(tiffs_fpaths) == 70
        
        return tiffs_fpaths


    # Method: Get a mapping of the IDs to the Full Path
    def get_id2fpath_wsi(self):

        id2fpath_wsi = dict()

        for tpath in self.tiffs_fpaths:
            fname = tpath.split('/')[-1].split('.')[0]
            # fname = f"{fname}_proc.tiff"
            fname = fname.replace('&', '_')
            fname = fname.replace(' ', '_')
            fname = fname.replace('(', '_')
            fname = fname.replace(')', '_')
            id2fpath_wsi[fname] = tpath

        return id2fpath_wsi


    # Method: Get the list of .pt files in the features directory
    def get_features_h5(self):

        features_h5_files = list()

        f_dir_folders = [f for f in os.listdir(self.features_h5_dir) if os.path.isdir(os.path.join(self.features_h5_dir, f))]            
        for folder in f_dir_folders:
            folder_files = [f for f in os.listdir(os.path.join(self.features_h5_dir, folder)) if not f.startswith('.')]
            if 'original.h5' in folder_files:
                features_h5_files.append(os.path.join(self.features_h5_dir, folder, 'original.h5'))
        # print(len(features_h5_files))

        return features_h5_files


    # Method: __len__
    def __len__(self):
        return len(self.features)


    # Method: __getitem__
    def __getitem__(self, idx):

        # Get features .PT file
        features_h5_path = self.features[idx]
        # print("features_h5_path: ", features_h5_path)
        with h5py.File(features_h5_path, "r") as f:
            features = f["features"][()]
        features = torch.from_numpy(features)
        # print("features.shape: ", features.shape)

        # Get WSI_ID
        wsi_id = features_h5_path.split('/')[-2].split('_proc')[0]
        # print("wsi_id: ", wsi_id)

        # Gat full path of the .TIFF
        fpath = self.id2fpath_wsi[wsi_id]

        # Build input dictionary
        input_data_dict = {
            'wsi_id':wsi_id,
            'fpath':fpath,
            'features_h5_path':features_h5_path,
            'features':features,
        }

        return input_data_dict



# Class: TCGABRCA_MIL_Dataset
class TCGABRCA_MIL_Dataset(Dataset):
    def __init__(self, base_data_path='TCGA-BRCA', experimental_strategy='All', label=None, features_h5_dir=None, train_size=0.70, val_size=0.15, test_size=0.15, n_folds=10, seed=42, transform=None):
        
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

        # Choose Experimental Strategy
        if experimental_strategy in ('DiagnosticSlide', 'TissueSlide'):
            if experimental_strategy == 'DiagnosticSlide':
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'DiagnosticSlide.csv'))
                assert len(data) == 1133, f'The size of the Diagnostic Slide subset should be 1133 and not {len(data)}.'
            else:
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'TissueSlide.csv'))
                assert len(data) == 1978, f'The size of the Tissue Slide subset should be 1978 and not {len(data)}.'
            
            # Get folders and SVS files
            folders, svs_files = data['folder_name'].values, data['svs_name']

            # Create a list for the SVS filepaths
            svs_fpaths = list()

            # Go through each folder subset and append the filepath to the list
            for folder, svs_file in zip(folders, svs_files):
                svs_fpath = os.path.join(base_data_path, 'WSI', folder, svs_file)
                svs_fpaths.append(svs_fpath)

        # Or get all data samples
        else:
            svs_fpaths = self.get_all_folders_and_svs_fpaths()


        # Get a dictionary of image paths according to their CIDs
        self.svs_fpaths = svs_fpaths
        self.svs_fpaths_dict = self.get_svs_fpaths_dict()

        # Get the .pt files with feature
        self.features = self.get_features_h5()

        # Get the ssGSEA Scores
        self.ssgsea_scores_dict = self.load_tcga_brca_ssgsea_scores()
        ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict = dict(), dict()
        
        for idx, label_ in enumerate(self.ssgsea_scores_dict['label_names']):
            ssgsea_scores_idx_label_dict[idx] = label_
            ssgsea_scores_label_idx_dict[label_] = idx
        
        self.ssgsea_scores_label_idx_dict, self.ssgsea_scores_idx_label_dict = ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict


        # Build dataset
        self.dataset_dict, self.wsi_genex_label_dict, self.features_h5_dict = self.build_dataset_dicts()


        # Apply train-val-test split according to the Case IDs
        self.train_dict, self.val_dict, self.test_dict = self.split_dataset()

        # Get label threshold
        self.label_threshold = 0

        # Class variables
        self.transform = transform
        
        return


    # Method: get_all_folders_and_svs
    def get_all_folders_and_svs_fpaths(self):

        # Enter the WSI directory of the dataset
        wsi_directory = os.path.join(self.base_data_path, 'WSI')

        # Get folders
        folders = [f for f in os.listdir(wsi_directory) if not f.startswith('.')]

        # Create a list for the SVS filepaths
        svs_fpaths = list()

        # Enter each folder
        for folder in folders:

            # Get the contents of each folder
            folder_contents = [c for c in os.listdir(os.path.join(wsi_directory, folder)) if not c.startswith('.')]
            
            # Get the SVS file(s)
            svs_files = [s for s in folder_contents if s.endswith('svs')]

            # Build SVS filepaths
            for svs_f in svs_files:
                svs_fpath = os.path.join(wsi_directory, folder, svs_f)

                # Append it to the list
                svs_fpaths.append(svs_fpath)

        return svs_fpaths


    # Method: Create a dictionary of svs filepaths connected to CIDs
    def get_svs_fpaths_dict(self):

        # Create a dictionary
        svs_fpaths_dict = dict()

        # Go through the svs_fpaths
        for path in self.svs_fpaths:
            
            # Get Case ID
            cid = self.get_case_id(wsi_path_or_name=path)

            # Add to dictionary
            if cid not in svs_fpaths_dict.keys():
                svs_fpaths_dict[cid] = list()

            svs_fpaths_dict[cid].append(path)

        return svs_fpaths_dict


    # Method: Get the list of .pt files in the features directory
    def get_features_h5(self):

        features_h5_files = list()

        for f_dir in self.features_h5_dir:
            f_dir_folders = [f for f in os.listdir(f_dir) if os.path.isdir(os.path.join(f_dir, f))]            
            for folder in f_dir_folders:
                folder_files = [f for f in os.listdir(os.path.join(f_dir, folder)) if not f.startswith('.')]
                if 'original.h5' in folder_files:
                    features_h5_files.append(os.path.join(f_dir, folder, 'original.h5'))
        # print(len(features_h5_files))

        return features_h5_files


    # Method: Load TCGA_BRCA_ssGSEA_Scores
    def load_tcga_brca_ssgsea_scores(self):

        # Read CSV
        df = pd.read_csv(os.path.join(self.base_data_path, 'Annotations', 'TCGA_BRCA_ssGSEA_Scores.csv'))
        
        # Create a data dictionary
        sgsea_scores_dict = dict()
        for col_name in list(df.columns):
            if col_name == 'Unnamed: 0':
                sgsea_scores_dict['label_names'] = df['Unnamed: 0'].values
            else:
                sgsea_scores_dict[col_name] = df[col_name].values

        return sgsea_scores_dict


    # Method: Obtain Case ID
    def get_case_id(self, wsi_path_or_name, mode='simple'):

        assert mode in ('simple', 'extended')

        # Pipeline to get Case ID
        parsed_path = wsi_path_or_name.split('/')[-1]
        parsed_path = parsed_path.split('.')[0]
        if mode == 'simple':
            parsed_path = parsed_path.split('-')[0:3]
        else:
            parsed_path = parsed_path.split('-')[0:4]

        # Get CID
        case_id = '-'.join(parsed_path)

        return case_id


    # Method: Build dataset dictionary
    def build_dataset_dicts(self):

        # Initialise dataset dictionary
        dataset_dict = {
            'case_id':list(),
            'svs_fpath':list(),
            'features_h5':list(),
            'ssgea_id':list(),
            'ssgsea_scores':list()
        }


        # Process the scores and get a dictionary that maps Case ID to column entry
        wsi_genex_label_dict = dict()
        for w in list(self.ssgsea_scores_dict.keys()):
            if w != 'label_names':
                case_id = self.get_case_id(wsi_path_or_name=w)
                if case_id not in wsi_genex_label_dict.keys():
                    wsi_genex_label_dict[case_id] = list()
                wsi_genex_label_dict[case_id].append(w)


        # Process the features names and obtain a dictionary that maps Case ID to filename
        features_h5_dict = dict()
        for f in self.features:
            case_id = self.get_case_id(wsi_path_or_name=f.split('/')[-2])
            if case_id not in features_h5_dict.keys():
                features_h5_dict[case_id] = list()
            features_h5_dict[case_id].append(f)



        # Process the WSIs
        for case_id in self.svs_fpaths_dict.keys():

            # Check if this Case ID is part of our annotations and features
            if case_id in wsi_genex_label_dict.keys() and case_id in features_h5_dict.keys():

                # Open all the paths in this case id
                for svs_path in self.svs_fpaths_dict[case_id]:

                    # print(svs_path)
                    # print(features_h5_dict[case_id])

                    # Obtain .h5 filename
                    wsi_fname = os.path.splitext(svs_path.split('/')[-1])[0]
                    feature_h5_fname = ''
                    for fname in features_h5_dict[case_id]:
                        if wsi_fname in fname.split('/'):
                            feature_h5_fname = fname
                            # print(wsi_fname)
                            # print(feature_h5_fname)

                    # Get the SSGEA scores
                    ssgea_scores_list = wsi_genex_label_dict[case_id]
                    case_id_ext = self.get_case_id(wsi_path_or_name=svs_path, mode='extended')

                    for ssgea in ssgea_scores_list:
                        ssgea_ext = self.get_case_id(wsi_path_or_name=ssgea, mode='extended')
                        if case_id_ext == ssgea_ext and feature_h5_fname in (features_h5_dict[case_id]):
                            dataset_dict['case_id'].append(case_id)
                            dataset_dict['svs_fpath'].append(svs_path)
                            dataset_dict['features_h5'].append(feature_h5_fname)
                            dataset_dict['ssgea_id'].append(ssgea)
                            dataset_dict['ssgsea_scores'].append(self.ssgsea_scores_dict[ssgea])


        # Ensure quality of the database
        assert len(dataset_dict['case_id']) == len(dataset_dict['svs_fpath'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['features_h5'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgea_id'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgsea_scores'])

        return dataset_dict, wsi_genex_label_dict, features_h5_dict


    # Method: Split dataset
    def split_dataset(self):
        
        # Initialize train, validation and test dictionaries
        trainval_dict, train_dict, val_dict, test_dict = dict(), dict(), dict(), dict()

        # Create for train-val & test
        gss_trainval_test = GroupShuffleSplit(
            n_splits=self.n_folds,
            train_size=self.train_size+self.val_size,
            random_state=self.seed
        )

        # Create for trai & val
        gss_train_val = GroupShuffleSplit(
            n_splits=1, 
            train_size=(self.train_size/(self.train_size+self.val_size)), 
            random_state=self.seed
        )

        # Split first into train-val & test
        groups = self.dataset_dict['case_id']
        X = self.dataset_dict['features_h5']
        y = self.dataset_dict['ssgsea_scores']
        for fold, (train_index, test_index) in enumerate(gss_trainval_test.split(X, y, groups)):
            trainval_dict[fold] = {
                'case_id':[self.dataset_dict['case_id'][i] for i in train_index],
                'svs_fpath':[self.dataset_dict['svs_fpath'][i] for i in train_index],
                'features_h5':[self.dataset_dict['features_h5'][i] for i in train_index],
                'ssgea_id':[self.dataset_dict['ssgea_id'][i] for i in train_index],
                'ssgsea_scores':[self.dataset_dict['ssgsea_scores'][i] for i in train_index]
            }
            test_dict[fold] = {
                'case_id':[self.dataset_dict['case_id'][i] for i in test_index],
                'svs_fpath':[self.dataset_dict['svs_fpath'][i] for i in test_index],
                'features_h5':[self.dataset_dict['features_h5'][i] for i in test_index],
                'ssgea_id':[self.dataset_dict['ssgea_id'][i] for i in test_index],
                'ssgsea_scores':[self.dataset_dict['ssgsea_scores'][i] for i in test_index]
            }
        

        # Split then into train & val
        for fold in range(self.n_folds):
            groups = trainval_dict[fold]['case_id']
            X = trainval_dict[fold]['features_h5']
            y = trainval_dict[fold]['ssgsea_scores']
            for _, (train_index, test_index) in enumerate(gss_train_val.split(X, y, groups)):
                train_dict[fold] = {
                    'case_id':[trainval_dict[fold]['case_id'][i] for i in train_index],
                    'svs_fpath':[trainval_dict[fold]['svs_fpath'][i] for i in train_index],
                    'features_h5':[trainval_dict[fold]['features_h5'][i] for i in train_index],
                    'ssgea_id':[trainval_dict[fold]['ssgea_id'][i] for i in train_index],
                    'ssgsea_scores':[trainval_dict[fold]['ssgsea_scores'][i] for i in train_index]
                }
                val_dict[fold] = {
                        'case_id':[trainval_dict[fold]['case_id'][i] for i in test_index],
                        'svs_fpath':[trainval_dict[fold]['svs_fpath'][i] for i in test_index],
                        'features_h5':[trainval_dict[fold]['features_h5'][i] for i in test_index],
                        'ssgea_id':[trainval_dict[fold]['ssgea_id'][i] for i in test_index],
                        'ssgsea_scores':[trainval_dict[fold]['ssgsea_scores'][i] for i in test_index]
                    }
        

        # Ensure quality of the database
        for fold in range(self.n_folds):
            assert len(self.dataset_dict['case_id']) == len(train_dict[fold]['case_id']) + len(val_dict[fold]['case_id']) + len(test_dict[fold]['case_id'])
            
            # train_ratio = len(train_dict[fold]['case_id']) / len(self.dataset_dict['case_id'])
            # print(f"Train ratio: {train_ratio}")
            
            # validation_ratio = len(val_dict[fold]['case_id']) / len(self.dataset_dict['case_id'])
            # print(f"Validation ratio: {validation_ratio}")
            
            # test_ratio = len(test_dict[fold]['case_id']) / len(self.dataset_dict['case_id'])
            # print(f"Test ratio: {test_ratio}")


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


    # Method: __len__
    def __len__(self):
        if self.curr_split == 'train':
            len_ = len(self.train_dict[self.curr_fold]['case_id'])
        elif self.curr_split == 'validation':
            len_ = len(self.val_dict[self.curr_fold]['case_id'])
        else:
            len_  = len(self.test_dict[self.curr_fold]['case_id'])

        return len_


    # Method: __getitem__
    def __getitem__(self, idx):

        # Initialise dataset dictionary
        dataset_dict = {
            'case_id':list(),
            'svs_fpath':list(),
            'features_h5':list(),
            'ssgea_id':list(),
            'ssgsea_scores':list()
        }

        # Select the dataset dictionary
        if self.curr_split == 'train':
            dataset_dict = self.train_dict[self.curr_fold]
        elif self.curr_split == 'validation':
            dataset_dict = self.val_dict[self.curr_fold]
        else:
            dataset_dict = self.test_dict[self.curr_fold]
        
        # Get Case ID
        case_id = dataset_dict['case_id'][idx]

        # Get SVS path
        svs_path = dataset_dict['svs_fpath'][idx]

        # Get features .PT file
        features_h5 = dataset_dict['features_h5'][idx]
        with h5py.File(features_h5, "r") as f:
            features = f["features"][()]
        features = torch.from_numpy(features)
        # print(features.shape)

        # Get SSGEA scores
        ssgea_id = dataset_dict['ssgea_id'][idx]
        ssgsea_scores = dataset_dict['ssgsea_scores'][idx][self.ssgsea_scores_label_idx_dict[self.label]]
        ssgsea_scores = float(ssgsea_scores)
        ssgsea_scores = 1 if ssgsea_scores > self.label_threshold else 0

        # Build input dictionary
        input_data_dict = {
            'case_id':case_id,
            'svs_path':svs_path,
            'features_h5':features_h5,
            'features':features,
            'ssgea_id':ssgea_id,
            'ssgsea_scores':ssgsea_scores
        }

        return input_data_dict



# Class: TCGABRCA_MIL_DatasetRegression
class TCGABRCA_MIL_DatasetRegression(TCGABRCA_MIL_Dataset):

    # Method: __init__
    def __init__(self, base_data_path='TCGA-BRCA', experimental_strategy='All', label=None, features_h5_dir=None, train_size=0.7, val_size=0.15, test_size=0.15, n_folds=10, seed=42, transform=None):
        super().__init__(base_data_path, experimental_strategy, label, features_h5_dir, train_size, val_size, test_size, n_folds, seed, transform)

        return


    # Method: __getitem__
    def __getitem__(self, idx):

        # Initialise dataset dictionary
        dataset_dict = dict()

        # Select the dataset dictionary
        if self.curr_split == 'train':
            dataset_dict = self.train_dict[self.curr_fold]
        elif self.curr_split == 'validation':
            dataset_dict = self.val_dict[self.curr_fold]
        else:
            dataset_dict = self.test_dict[self.curr_fold]
        
        # Get Case ID
        case_id = dataset_dict['case_id'][idx]

        # Get SVS path
        svs_path = dataset_dict['svs_fpath'][idx]

        # Get features .PT file
        features_h5 = dataset_dict['features_h5'][idx]
        with h5py.File(features_h5, "r") as f:
            features = f["features"][()]
        features = torch.from_numpy(features)
        # print(features.shape)

        # Get SSGEA scores
        ssgea_id = dataset_dict['ssgea_id'][idx]
        ssgsea_scores = dataset_dict['ssgsea_scores'][idx][self.ssgsea_scores_label_idx_dict[self.label]]
        ssgsea_scores = float(ssgsea_scores)
        ssgsea_scores_bin = 1 if ssgsea_scores > self.label_threshold else 0

        # Build input dictionary
        input_data_dict = {
            'case_id':case_id,
            'svs_path':svs_path,
            'features_h5':features_h5,
            'features':features,
            'ssgea_id':ssgea_id,
            'ssgsea_scores':ssgsea_scores,
            'ssgsea_scores_bin':ssgsea_scores_bin
        }

        return input_data_dict



# Class: TCGABRCA_MIL_DatasetClinicalSubtype
class TCGABRCA_MIL_DatasetClinicalSubtype(Dataset):

    # Method: __init__
    def __init__(self, base_data_path='TCGA-BRCA', experimental_strategy='All', label=None, features_h5_dir=None, train_size=0.70, val_size=0.15, test_size=0.15, n_folds=10, seed=42, transform=None):
        
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

        # Choose Experimental Strategy
        if experimental_strategy in ('DiagnosticSlide', 'TissueSlide'):
            if experimental_strategy == 'DiagnosticSlide':
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'DiagnosticSlide.csv'))
                assert len(data) == 1133, f'The size of the Diagnostic Slide subset should be 1133 and not {len(data)}.'
            else:
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'TissueSlide.csv'))
                assert len(data) == 1978, f'The size of the Tissue Slide subset should be 1978 and not {len(data)}.'
            
            # Get folders and SVS files
            folders, svs_files = data['folder_name'].values, data['svs_name']

            # Create a list for the SVS filepaths
            svs_fpaths = list()

            # Go through each folder subset and append the filepath to the list
            for folder, svs_file in zip(folders, svs_files):
                svs_fpath = os.path.join(base_data_path, 'WSI', folder, svs_file)
                svs_fpaths.append(svs_fpath)

        # Or get all data samples
        else:
            svs_fpaths = self.get_all_folders_and_svs_fpaths()


        # Get a dictionary of image paths according to their CIDs
        self.svs_fpaths = svs_fpaths
        self.svs_fpaths_dict = self.get_svs_fpaths_dict()

        # Get the .pt files with feature
        self.features = self.get_features_h5()

        # Get the ssGSEA Scores
        self.ssgsea_scores_dict = self.load_tcga_brca_ssgsea_scores()
        ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict = dict(), dict()
        
        for idx, label_ in enumerate(self.ssgsea_scores_dict['label_names']):
            ssgsea_scores_idx_label_dict[idx] = label_
            ssgsea_scores_label_idx_dict[label_] = idx
        
        self.ssgsea_scores_label_idx_dict, self.ssgsea_scores_idx_label_dict = ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict

        # Get the Clinical Subtype
        self.tcga_brca_labels_dict = self.load_tcga_brca_labels()
        # print(self.tcga_brca_labels_dict)
        tcga_brca_label_idx_dict = {
            "HER2+/HR+":0,
            "HER2+/HR-":1,
            "HER2-/HR+":2,
            "HER2-/HR-":3
        }
        tcga_brca_idx_label_dict = dict()
        for k, v in tcga_brca_label_idx_dict.items():
            tcga_brca_idx_label_dict[v] = k
        self.tcga_brca_label_idx_dict, self.tcga_brca_idx_label_dict = tcga_brca_label_idx_dict, tcga_brca_idx_label_dict
        

        # Build dataset
        self.dataset_dict, self.wsi_genex_label_dict, self.features_h5_dict = self.build_dataset_dicts()


        # Apply train-val-test split according to the Case IDs
        self.train_dict, self.val_dict, self.test_dict = self.split_dataset()

        # Get label threshold
        self.label_threshold = 0

        # Class variables
        self.transform = transform
        
        return


    # Method: get_all_folders_and_svs
    def get_all_folders_and_svs_fpaths(self):

        # Enter the WSI directory of the dataset
        wsi_directory = os.path.join(self.base_data_path, 'WSI')

        # Get folders
        folders = [f for f in os.listdir(wsi_directory) if not f.startswith('.')]

        # Create a list for the SVS filepaths
        svs_fpaths = list()

        # Enter each folder
        for folder in folders:

            # Get the contents of each folder
            folder_contents = [c for c in os.listdir(os.path.join(wsi_directory, folder)) if not c.startswith('.')]
            
            # Get the SVS file(s)
            svs_files = [s for s in folder_contents if s.endswith('svs')]

            # Build SVS filepaths
            for svs_f in svs_files:
                svs_fpath = os.path.join(wsi_directory, folder, svs_f)

                # Append it to the list
                svs_fpaths.append(svs_fpath)

        return svs_fpaths


    # Method: Create a dictionary of svs filepaths connected to CIDs
    def get_svs_fpaths_dict(self):

        # Create a dictionary
        svs_fpaths_dict = dict()

        # Go through the svs_fpaths
        for path in self.svs_fpaths:
            
            # Get Case ID
            cid = self.get_case_id(wsi_path_or_name=path)

            # Add to dictionary
            if cid not in svs_fpaths_dict.keys():
                svs_fpaths_dict[cid] = list()

            svs_fpaths_dict[cid].append(path)

        return svs_fpaths_dict


    # Method: Get the list of .pt files in the features directory
    def get_features_h5(self):

        features_h5_files = list()

        for f_dir in self.features_h5_dir:
            f_dir_folders = [f for f in os.listdir(f_dir) if os.path.isdir(os.path.join(f_dir, f))]            
            for folder in f_dir_folders:
                folder_files = [f for f in os.listdir(os.path.join(f_dir, folder)) if not f.startswith('.')]
                if 'original.h5' in folder_files:
                    features_h5_files.append(os.path.join(f_dir, folder, 'original.h5'))
        # print(len(features_h5_files))

        return features_h5_files


    # Method: Load TCGA_BRCA_ssGSEA_Scores
    def load_tcga_brca_ssgsea_scores(self):

        # Read CSV
        df = pd.read_csv(os.path.join(self.base_data_path, 'Annotations', 'TCGA_BRCA_ssGSEA_Scores.csv'))
        
        # Create a data dictionary
        sgsea_scores_dict = dict()
        for col_name in list(df.columns):
            if col_name == 'Unnamed: 0':
                sgsea_scores_dict['label_names'] = df['Unnamed: 0'].values
            else:
                sgsea_scores_dict[col_name] = df[col_name].values

        return sgsea_scores_dict
    

    # Method: Load TCGA-BRCA Labels
    def load_tcga_brca_labels(self):

        # Load the labels
        tcga_brca_labels = pd.read_excel(os.path.join(self.base_data_path, 'Annotations', 'breast_cancer_tcga_labels_1.xlsx'))
        tcga_brca_labels = tcga_brca_labels.copy()[['CLID', 'er_status_by_ihc', 'pr_status_by_ihc', 'HER2.newly.derived']]

        tcga_brca_labels_dict = dict()

        for _, r in tcga_brca_labels.iterrows():
            clid = self.get_case_id(r['CLID'])
            er = r['er_status_by_ihc']
            pr = r['pr_status_by_ihc']
            her2 = r['HER2.newly.derived']
            c_subtype, c_subtype_label = self.compute_clinical_subtype(er, pr, her2)
            if c_subtype is not None:
                if clid not in tcga_brca_labels_dict.keys():
                    tcga_brca_labels_dict[clid] = dict()
                tcga_brca_labels_dict[clid]["c_subtype"] = c_subtype
                tcga_brca_labels_dict[clid]["c_subtype_label"] = c_subtype_label
                tcga_brca_labels_dict[clid]["er"] = er
                tcga_brca_labels_dict[clid]["pr"] = pr
                tcga_brca_labels_dict[clid]["her2"] = her2

        return tcga_brca_labels_dict


    # Method: Obtain Case ID
    def get_case_id(self, wsi_path_or_name, mode='simple'):

        assert mode in ('simple', 'extended')

        # Pipeline to get Case ID
        parsed_path = wsi_path_or_name.split('/')[-1]
        parsed_path = parsed_path.split('.')[0]
        if mode == 'simple':
            parsed_path = parsed_path.split('-')[0:3]
        else:
            parsed_path = parsed_path.split('-')[0:4]

        # Get CID
        case_id = '-'.join(parsed_path)

        return case_id


    # Function: Compute clinical subtype
    def compute_clinical_subtype(self, er, pr, her2):

        # Source: https://www.cancer.org/cancer/types/breast-cancer/understanding-a-breast-cancer-diagnosis/breast-cancer-hormone-receptor-status.html
        # Clinical Subtypes: HER+/HR+, HER+/HR-, HER-/HR+, HER-/HR- (triple negative)
        # Define HR = ER + PR
        # HR+ <-> ER+ or PR+
        # HR- <-> ER- and PR-

        
        # Compute HR
        hr = None
        if er == "Positive" or pr == "Positive":
            hr = "Positive"
        elif er == "Negative" and pr == "Negative":
            hr = "Negative"


        # Compute Clinical Subtype
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


    # Method: Build dataset dictionary
    def build_dataset_dicts(self):

        # Initialise dataset dictionary
        dataset_dict = {
            'case_id':list(),
            'svs_fpath':list(),
            'features_h5':list(),
            'ssgea_id':list(),
            'ssgsea_scores':list(),
            'c_subtype':list(),
            'c_subtype_label':list()
        }


        # Process the scores and get a dictionary that maps Case ID to column entry
        wsi_genex_label_dict = dict()
        for w in list(self.ssgsea_scores_dict.keys()):
            if w != 'label_names':
                case_id = self.get_case_id(wsi_path_or_name=w)
                if case_id not in wsi_genex_label_dict.keys():
                    wsi_genex_label_dict[case_id] = list()
                wsi_genex_label_dict[case_id].append(w)


        # Process the features names and obtain a dictionary that maps Case ID to filename
        features_h5_dict = dict()
        for f in self.features:
            case_id = self.get_case_id(wsi_path_or_name=f.split('/')[-2])
            if case_id not in features_h5_dict.keys():
                features_h5_dict[case_id] = list()
            features_h5_dict[case_id].append(f)



        # Process the WSIs
        for case_id in self.svs_fpaths_dict.keys():

            # Check if this Case ID is part of our annotations and features
            if case_id in wsi_genex_label_dict.keys() and case_id in features_h5_dict.keys():

                # Open all the paths in this case id
                for svs_path in self.svs_fpaths_dict[case_id]:

                    # print(svs_path)
                    # print(features_h5_dict[case_id])

                    # Obtain .h5 filename
                    wsi_fname = os.path.splitext(svs_path.split('/')[-1])[0]
                    feature_h5_fname = ''
                    for fname in features_h5_dict[case_id]:
                        if wsi_fname in fname.split('/'):
                            feature_h5_fname = fname
                            # print(wsi_fname)
                            # print(feature_h5_fname)

                    # Get the SSGEA scores
                    ssgea_scores_list = wsi_genex_label_dict[case_id]
                    case_id_ext = self.get_case_id(wsi_path_or_name=svs_path, mode='extended')

                    for ssgea in ssgea_scores_list:
                        ssgea_ext = self.get_case_id(wsi_path_or_name=ssgea, mode='extended')
                        if case_id_ext == ssgea_ext and feature_h5_fname in (features_h5_dict[case_id]) and case_id in self.tcga_brca_labels_dict.keys():
                            dataset_dict['case_id'].append(case_id)
                            dataset_dict['svs_fpath'].append(svs_path)
                            dataset_dict['features_h5'].append(feature_h5_fname)
                            dataset_dict['ssgea_id'].append(ssgea)
                            dataset_dict['ssgsea_scores'].append(self.ssgsea_scores_dict[ssgea])
                            dataset_dict['c_subtype'].append(self.tcga_brca_labels_dict[case_id]['c_subtype'])
                            dataset_dict['c_subtype_label'].append(self.tcga_brca_labels_dict[case_id]['c_subtype_label'])


        # Ensure quality of the database
        assert len(dataset_dict['case_id']) == len(dataset_dict['svs_fpath'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['features_h5'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgea_id'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgsea_scores'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['c_subtype'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['c_subtype_label'])
        

        return dataset_dict, wsi_genex_label_dict, features_h5_dict


    # Method: Split dataset
    def split_dataset(self):
        
        # Initialize train, validation and test dictionaries
        trainval_dict, train_dict, val_dict, test_dict = dict(), dict(), dict(), dict()

        # Create for train-val & test
        gss_trainval_test = GroupShuffleSplit(
            n_splits=self.n_folds,
            train_size=self.train_size+self.val_size,
            random_state=self.seed
        )

        # Create for trai & val
        gss_train_val = GroupShuffleSplit(
            n_splits=1, 
            train_size=(self.train_size/(self.train_size+self.val_size)), 
            random_state=self.seed
        )

        # Split first into train-val & test
        groups = self.dataset_dict['case_id']
        X = self.dataset_dict['features_h5']
        y = self.dataset_dict['ssgsea_scores']
        for fold, (train_index, test_index) in enumerate(gss_trainval_test.split(X, y, groups)):
            trainval_dict[fold] = {
                'case_id':[self.dataset_dict['case_id'][i] for i in train_index],
                'svs_fpath':[self.dataset_dict['svs_fpath'][i] for i in train_index],
                'features_h5':[self.dataset_dict['features_h5'][i] for i in train_index],
                'ssgea_id':[self.dataset_dict['ssgea_id'][i] for i in train_index],
                'ssgsea_scores':[self.dataset_dict['ssgsea_scores'][i] for i in train_index],
                'c_subtype':[self.dataset_dict['c_subtype'][i] for i in train_index],
                'c_subtype_label':[self.dataset_dict['c_subtype_label'][i] for i in train_index]
            }
            test_dict[fold] = {
                'case_id':[self.dataset_dict['case_id'][i] for i in test_index],
                'svs_fpath':[self.dataset_dict['svs_fpath'][i] for i in test_index],
                'features_h5':[self.dataset_dict['features_h5'][i] for i in test_index],
                'ssgea_id':[self.dataset_dict['ssgea_id'][i] for i in test_index],
                'ssgsea_scores':[self.dataset_dict['ssgsea_scores'][i] for i in test_index],
                'c_subtype':[self.dataset_dict['c_subtype'][i] for i in test_index],
                'c_subtype_label':[self.dataset_dict['c_subtype_label'][i] for i in test_index]
            }
        

        # Split then into train & val
        for fold in range(self.n_folds):
            groups = trainval_dict[fold]['case_id']
            X = trainval_dict[fold]['features_h5']
            y = trainval_dict[fold]['ssgsea_scores']
            for _, (train_index, test_index) in enumerate(gss_train_val.split(X, y, groups)):
                train_dict[fold] = {
                    'case_id':[trainval_dict[fold]['case_id'][i] for i in train_index],
                    'svs_fpath':[trainval_dict[fold]['svs_fpath'][i] for i in train_index],
                    'features_h5':[trainval_dict[fold]['features_h5'][i] for i in train_index],
                    'ssgea_id':[trainval_dict[fold]['ssgea_id'][i] for i in train_index],
                    'ssgsea_scores':[trainval_dict[fold]['ssgsea_scores'][i] for i in train_index],
                    'c_subtype':[trainval_dict[fold]['c_subtype'][i] for i in train_index],
                    'c_subtype_label':[trainval_dict[fold]['c_subtype_label'][i] for i in train_index]
                }
                val_dict[fold] = {
                        'case_id':[trainval_dict[fold]['case_id'][i] for i in test_index],
                        'svs_fpath':[trainval_dict[fold]['svs_fpath'][i] for i in test_index],
                        'features_h5':[trainval_dict[fold]['features_h5'][i] for i in test_index],
                        'ssgea_id':[trainval_dict[fold]['ssgea_id'][i] for i in test_index],
                        'ssgsea_scores':[trainval_dict[fold]['ssgsea_scores'][i] for i in test_index],
                        'c_subtype':[trainval_dict[fold]['c_subtype'][i] for i in test_index],
                        'c_subtype_label':[trainval_dict[fold]['c_subtype_label'][i] for i in test_index]
                    }
        

        # Ensure quality of the database
        for fold in range(self.n_folds):
            assert len(self.dataset_dict['case_id']) == len(train_dict[fold]['case_id']) + len(val_dict[fold]['case_id']) + len(test_dict[fold]['case_id'])
            
            # train_ratio = len(train_dict[fold]['case_id']) / len(self.dataset_dict['case_id'])
            # print(f"Train ratio: {train_ratio}")
            
            # validation_ratio = len(val_dict[fold]['case_id']) / len(self.dataset_dict['case_id'])
            # print(f"Validation ratio: {validation_ratio}")
            
            # test_ratio = len(test_dict[fold]['case_id']) / len(self.dataset_dict['case_id'])
            # print(f"Test ratio: {test_ratio}")


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


    # Method: __len__
    def __len__(self):
        if self.curr_split == 'train':
            len_ = len(self.train_dict[self.curr_fold]['case_id'])
        elif self.curr_split == 'validation':
            len_ = len(self.val_dict[self.curr_fold]['case_id'])
        else:
            len_  = len(self.test_dict[self.curr_fold]['case_id'])

        return len_


    # Method: __getitem__
    def __getitem__(self, idx):

        # Initialise dataset dictionary
        dataset_dict = {
            'case_id':list(),
            'svs_fpath':list(),
            'features_h5':list(),
            'ssgea_id':list(),
            'ssgsea_scores':list(),
            'c_subtype':list(),
            'c_subtype_label':list()
        }

        # Select the dataset dictionary
        if self.curr_split == 'train':
            dataset_dict = self.train_dict[self.curr_fold]
        elif self.curr_split == 'validation':
            dataset_dict = self.val_dict[self.curr_fold]
        else:
            dataset_dict = self.test_dict[self.curr_fold]
        
        # Get Case ID
        case_id = dataset_dict['case_id'][idx]

        # Get SVS path
        svs_path = dataset_dict['svs_fpath'][idx]

        # Get features .PT file
        features_h5 = dataset_dict['features_h5'][idx]
        with h5py.File(features_h5, "r") as f:
            features = f["features"][()]
        features = torch.from_numpy(features)
        # print(features.shape)

        # Get SSGEA scores
        ssgea_id = dataset_dict['ssgea_id'][idx]
        if self.label:
            ssgsea_scores = dataset_dict['ssgsea_scores'][idx][self.ssgsea_scores_label_idx_dict[self.label]]
            ssgsea_scores = float(ssgsea_scores)
            ssgsea_scores = 1 if ssgsea_scores > self.label_threshold else 0
        else:
            ssgsea_scores = -1
        
        # Get Clinical Subtype and Clinical Subtype Label
        c_subtype = dataset_dict['c_subtype'][idx]
        c_subtype_label = dataset_dict['c_subtype_label'][idx]

        # Build input dictionary
        input_data_dict = {
            'case_id':case_id,
            'svs_path':svs_path,
            'features_h5':features_h5,
            'features':features,
            'ssgea_id':ssgea_id,
            'ssgsea_scores':ssgsea_scores,
            'c_subtype':c_subtype,
            'c_subtype_label':c_subtype_label
        }

        return input_data_dict