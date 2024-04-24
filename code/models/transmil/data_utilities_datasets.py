# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Imports
import torch
from torch.utils.data import Dataset

# Sklearn Imports
from sklearn.model_selection import GroupShuffleSplit



# Class: TCGABRCA_MIL_Dataset
class TCGABRCA_MIL_Dataset(Dataset):
    def __init__(self, base_data_path='TCGA-BRCA', experimental_strategy='All', setting='binary', label=None, label_thresh_metric='zero', features_pt_dir=None, train_size=0.70, val_size=0.15, test_size=0.15, n_folds=10, seed=42, transform=None):
        
        assert experimental_strategy in ('All', 'DiagnosticSlide', 'TissueSlide')
        assert features_pt_dir is not None
        assert setting in ('binary')
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
        assert label_thresh_metric in ('average', 'median', 'zero')
        assert train_size + val_size + test_size == 1
        assert n_folds > 0
        assert seed is not None

        # Class variables
        self.base_data_path = base_data_path
        self.experimental_strategy = experimental_strategy
        self.features_pt_dir = features_pt_dir
        self.setting = setting
        self.label = label
        self.label_thresh_metric = label_thresh_metric
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
        self.features = self.get_features_pt()

        # Get the ssGSEA Scores
        self.ssgsea_scores_dict = self.load_tcga_brca_ssgsea_scores()
        ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict = dict(), dict()
        
        for idx, label_ in enumerate(self.ssgsea_scores_dict['label_names']):
            ssgsea_scores_idx_label_dict[idx] = label_
            ssgsea_scores_label_idx_dict[label_] = idx
        
        self.ssgsea_scores_label_idx_dict, self.ssgsea_scores_idx_label_dict = ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict


        # Build dataset
        self.dataset_dict, self.wsi_genex_label_dict, self.features_pt_dict = self.build_dataset_dicts()
        self.update_features_pt_paths()

        # Apply train-val-test split according to the Case IDs
        self.train_dict, self.val_dict, self.test_dict = self.split_dataset()

        # Get label threshold
        self.label_threshold = self.get_label_threshold() if self.label_thresh_metric in ('average', 'median') else 0

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
    def get_features_pt(self):

        features_pt_files = list()

        for f_dir in self.features_pt_dir:
            f_dir_pt_files = [f for f in os.listdir(f_dir) if not f.startswith('.')]
            features_pt_files += f_dir_pt_files

        return features_pt_files


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
            'features_pt':list(),
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
        features_pt_dict = dict()
        for f in self.features:
            case_id = self.get_case_id(wsi_path_or_name=f)
            if case_id not in features_pt_dict.keys():
                features_pt_dict[case_id] = list()
            features_pt_dict[case_id].append(f)



        # Process the WSIs
        for case_id in self.svs_fpaths_dict.keys():
            
            # Check if this Case ID is part of our annotations and features
            if case_id in wsi_genex_label_dict.keys() and case_id in features_pt_dict.keys():

                # Open all the paths in this case id
                for svs_path in self.svs_fpaths_dict[case_id]:
                    
                    

                    # Obtain .PT filename
                    feature_pt_fname = svs_path.split('/')[-1]
                    feature_pt_fname = feature_pt_fname[0:-4] + '.pt'

                    # Get the SSGEA scores
                    ssgea_scores_list = wsi_genex_label_dict[case_id]
                    case_id_ext = self.get_case_id(wsi_path_or_name=svs_path, mode='extended')

                    for ssgea in ssgea_scores_list:
                        ssgea_ext = self.get_case_id(wsi_path_or_name=ssgea, mode='extended')
                        if case_id_ext == ssgea_ext and feature_pt_fname in (features_pt_dict[case_id]):
                            dataset_dict['case_id'].append(case_id)
                            dataset_dict['svs_fpath'].append(svs_path)
                            dataset_dict['features_pt'].append(feature_pt_fname)
                            dataset_dict['ssgea_id'].append(ssgea)
                            dataset_dict['ssgsea_scores'].append(self.ssgsea_scores_dict[ssgea])


        # Ensure quality of the database
        assert len(dataset_dict['case_id']) == len(dataset_dict['svs_fpath'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['features_pt'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgea_id'])
        assert len(dataset_dict['case_id']) == len(dataset_dict['ssgsea_scores'])

        return dataset_dict, wsi_genex_label_dict, features_pt_dict


    # Method: Update paths of the features .PT files
    def update_features_pt_paths(self):

        # Get all the features .PT files
        for idx, fpt_fname in enumerate(self.dataset_dict['features_pt']):
            for fpt_dir in self.features_pt_dir:
                fpt_fpath = os.path.join(fpt_dir, fpt_fname)
                if os.path.exists(fpt_fpath):
                    self.dataset_dict['features_pt'][idx] = fpt_fpath

        return


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
        X = self.dataset_dict['features_pt']
        y = self.dataset_dict['ssgsea_scores']
        for fold, (train_index, test_index) in enumerate(gss_trainval_test.split(X, y, groups)):
            trainval_dict[fold] = {
                'case_id':[self.dataset_dict['case_id'][i] for i in train_index],
                'svs_fpath':[self.dataset_dict['svs_fpath'][i] for i in train_index],
                'features_pt':[self.dataset_dict['features_pt'][i] for i in train_index],
                'ssgea_id':[self.dataset_dict['ssgea_id'][i] for i in train_index],
                'ssgsea_scores':[self.dataset_dict['ssgsea_scores'][i] for i in train_index]
            }
            test_dict[fold] = {
                'case_id':[self.dataset_dict['case_id'][i] for i in test_index],
                'svs_fpath':[self.dataset_dict['svs_fpath'][i] for i in test_index],
                'features_pt':[self.dataset_dict['features_pt'][i] for i in test_index],
                'ssgea_id':[self.dataset_dict['ssgea_id'][i] for i in test_index],
                'ssgsea_scores':[self.dataset_dict['ssgsea_scores'][i] for i in test_index]
            }
        

        # Split then into train & val
        for fold in range(self.n_folds):
            groups = trainval_dict[fold]['case_id']
            X = trainval_dict[fold]['features_pt']
            y = trainval_dict[fold]['ssgsea_scores']
            for _, (train_index, test_index) in enumerate(gss_train_val.split(X, y, groups)):
                train_dict[fold] = {
                    'case_id':[trainval_dict[fold]['case_id'][i] for i in train_index],
                    'svs_fpath':[trainval_dict[fold]['svs_fpath'][i] for i in train_index],
                    'features_pt':[trainval_dict[fold]['features_pt'][i] for i in train_index],
                    'ssgea_id':[trainval_dict[fold]['ssgea_id'][i] for i in train_index],
                    'ssgsea_scores':[trainval_dict[fold]['ssgsea_scores'][i] for i in train_index]
                }
                val_dict[fold] = {
                        'case_id':[trainval_dict[fold]['case_id'][i] for i in test_index],
                        'svs_fpath':[trainval_dict[fold]['svs_fpath'][i] for i in test_index],
                        'features_pt':[trainval_dict[fold]['features_pt'][i] for i in test_index],
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


    # Method: Get label threshold for binary classification
    def get_label_threshold(self):

        # Get the label index
        label_idx = self.ssgsea_scores_label_idx_dict[self.label]
        label_scores = list()

        for sample in list(self.ssgsea_scores_dict.keys()):
            if sample != 'label_names':
                score = ssgsea_scores_dict[sample][label_idx]
                label_scores.append(score)
        
        label_threshold = np.average(label_scores) if self.label_thresh_metric == 'average' else np.median(label_scores)

        return label_threshold


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
            'features_pt':list(),
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
        features_pt = dataset_dict['features_pt'][idx]
        features = torch.load(os.path.join(features_pt))

        # Get SSGEA scores
        ssgea_id = dataset_dict['ssgea_id'][idx]
        ssgsea_scores = dataset_dict['ssgsea_scores'][idx][self.ssgsea_scores_label_idx_dict[self.label]]
        ssgsea_scores = float(ssgsea_scores)
        if self.setting == 'binary':
            ssgsea_scores = 1 if ssgsea_scores > self.label_threshold else 0

        # Build input dictionary
        input_data_dict = {
            'case_id':case_id,
            'svs_path':svs_path,
            'features_pt':features_pt,
            'features':features,
            'ssgea_id':ssgea_id,
            'ssgsea_scores':ssgsea_scores
        }

        return input_data_dict



# Usage
if __name__ == "__main__":

    # Imports
    import argparse

    # CLI Arguments
    parser = argparse.ArgumentParser(prog='TCGA-BRCA Dataset.', description='Script to test the TCGA-BRCA Dataset class.')
    parser.add_argument('--base_data_path', type=str, required=True, help="The path to the TCGA-BRCA database directory.")
    parser.add_argument('--experimental_strategy', type=str, required=True, choices=['All', 'DiagnosticSlide', 'TissueSlide'], help="The Experiemental Strategy subset.")
    parser.add_argument('--features_pt_dir', type=str, required=True, help="The path to the TCGA-BRCA features directory (contains .pt files).")
    parser.add_argument('--results_dir', type=str, help="The path to the directory of results.")
    args = parser.parse_args()

    # Create the dataset
    d = TCGABRCA_MIL_Dataset(
        base_data_path=args.base_data_path,
        experimental_strategy=args.experimental_strategy,
        features_pt_dir=args.features_pt_dir
    )

    # Get the SSGEA Scores Annotations
    ssgsea_scores_dict = d.ssgsea_scores_dict
    ssgsea_scores_label_idx_dict = d.ssgsea_scores_label_idx_dict

    eda_dict = dict()
    for label in ssgsea_scores_dict['label_names']:
        label_idx = ssgsea_scores_label_idx_dict[label]
        
        eda_dict[label] = list()

        for sample in list(ssgsea_scores_dict.keys()):
            if sample != 'label_names':
                score = ssgsea_scores_dict[sample][label_idx]
                eda_dict[label].append(score)
    
    for label, values in eda_dict.items():
        print(f"Label: {label} | Min: {np.min(values)}, Max: {np.max(values)}, Avg: {np.average(values)}")
        plt.title(f"Histogram | {label}")
        plt.hist(values, density=False)
        plt.axvline(np.average(values), color='red', linestyle='--', linewidth=3, label='Avg')
        plt.axvline(np.median(values), color='green', linestyle='--', linewidth=3, label='Med')
        plt.legend()
        plt.savefig(
            fname=os.path.join(args.results_dir, f'hist_{label}.png'),
            bbox_inches='tight'
        )
        plt.show()
        plt.close()

        plt.title(f"Distribution | {label}")
        plt.plot(values, 'bo')
        plt.axhline(np.average(values), color='red', linestyle='--', linewidth=3, label='Avg')
        plt.axhline(np.median(values), color='green', linestyle='--', linewidth=3, label='Med')
        plt.legend()
        plt.savefig(
            fname=os.path.join(args.results_dir, f'distribution_{label}.png'),
            bbox_inches='tight'
        )
        plt.show()
        plt.close()
