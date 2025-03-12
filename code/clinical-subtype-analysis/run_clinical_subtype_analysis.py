# Imports
from __future__ import print_function
import os
import argparse
import pandas as pd
import numpy as np
import random
import json
import copy
import matplotlib.pyplot as plt

# PyTorch Imports
import torch

# Project Imports
from train_val_test_utilities import test_pipeline
from data_utilities import TCGABRCA_MIL_DatasetClinicalSubtype



# Function: See the seed for reproducibility purposes
def set_seed(seed=42):

    # Random Seed
    random.seed(seed)

    # Environment Variable Seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy Seed
    np.random.seed(seed)

    # PyTorch Seed(s)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return



if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(description='CLAM: Clinical Subtype Analysis after Inference.')
    parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU we will use to run the program.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='The path to the checkpoint directory.')
    parser.add_argument('--dataset', type=str, required=True, choices=['TCGA-BRCA'], help='The dataset for the experiments.')
    parser.add_argument('--base_data_path', type=str, required=True, help='Base data path for TCGA-BRCA dataset.')
    parser.add_argument('--experimental_strategy', type=str, choices=['All', 'DiagnosticSlide', 'TissueSlide'], required=True, help="The experimental strategy for the TCGA-BRCA dataset.")
    parser.add_argument('--features_h5_dir', nargs='+', type=str, required=True, help="The directory of the features in .pt format for the TCGA-BRCA dataset.")
    args = parser.parse_args()



    # Set seeds (for reproducibility reasons)
    set_seed(seed=args.seed)


    # Load configuration JSON
    with open(os.path.join(args.checkpoint_dir, "config.json"), 'r') as j:
        config_json = json.load(j)


    # Load GPU/CPU device
    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')


    # Get the encoding size for the feature vectors
    encoding_size = config_json['data']['encoding_size']

    # Get verbose
    verbose = config_json['verbose']

    # Task type
    if "task_type" in config_json.keys():
        task_type = config_json["task_type"]
    else:
        task_type = "classification"
        config_json["task_type"] = "classification"

    
    # Load data
    print('Loading dataset...')
    label = args.checkpoint_dir.split('/')[-2]
    if args.dataset == 'TCGA-BRCA':
        dataset = TCGABRCA_MIL_DatasetClinicalSubtype(
            base_data_path=args.base_data_path,
            experimental_strategy=args.experimental_strategy,
            label=label,
            features_h5_dir=args.features_h5_dir,
            n_folds=int(config_json["data"]["n_folds"]),
            seed=int(args.seed)
        )


        # Create the data splits from the original dataset
        train_set = copy.deepcopy(dataset)
        train_set.select_split(split='train')

        val_set = copy.deepcopy(dataset)
        val_set.select_split(split='validation')

        test_set = copy.deepcopy(dataset)
        test_set.select_split(split='test')


    # Iterate through folds
    n_folds = int(config_json["data"]["n_folds"])
    for fold in range(n_folds):

        # Set seed
        set_seed(seed=args.seed)

        if verbose:
            print(f"Current Fold {fold+1}/{n_folds}")
        

        # Select folds in the database
        train_set.select_fold(fold=fold),
        val_set.select_fold(fold=fold),
        test_set.select_fold(fold=fold)

        for eval_set, eval_name in zip((train_set, val_set, test_set),('train', 'val', 'test')):
            test_inference_info = test_pipeline(
                test_set=eval_set,
                config_json=config_json,
                device=device,
                checkpoint_dir=args.checkpoint_dir,
                fold=fold
            )

            # Convert test metrics into a dataframe
            test_inference_info_df = pd.DataFrame.from_dict(test_inference_info)
            test_inference_info_df.to_csv(os.path.join(args.checkpoint_dir, f"{eval_name}_inference_info_kf{fold}.csv"))
            os.makedirs(os.path.join('results', label), exist_ok=True)
            test_inference_info_df.to_csv(os.path.join('results', label, f"{eval_name}_inference_info_kf{fold}.csv"))
            # print(eval_name)
            # print(test_inference_info_df)