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
from data_utilities import TCGABRCA_MIL_Dataset, TCGABRCA_MIL_DatasetRegression, TCGABRCA_MIL_DatasetClinicalSubtype



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
    parser = argparse.ArgumentParser(description='CLAM: Model Training.')
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
    task_type = config_json["task_type"]

    
    # Load data
    print('Loading dataset...')
    if args.dataset == 'TCGA-BRCA':
        if task_type == "classification":
            dataset = TCGABRCA_MIL_Dataset(
                base_data_path=args.base_data_path,
                experimental_strategy=args.experimental_strategy,
                label=args.checkpoint_dir.split('/')[-2],
                features_h5_dir=args.features_h5_dir,
                n_folds=int(config_json["data"]["n_folds"]),
                seed=int(args.seed)
            )
        elif task_type == "clinical_subtype_classification":
            dataset = TCGABRCA_MIL_DatasetClinicalSubtype(
                base_data_path=args.base_data_path,
                experimental_strategy=args.experimental_strategy,
                features_h5_dir=args.features_h5_dir,
                n_folds=int(config_json["data"]["n_folds"]),
                seed=int(args.seed)
            )
        elif task_type == "regression":
            dataset = TCGABRCA_MIL_DatasetRegression(
                base_data_path=args.base_data_path,
                    experimental_strategy=args.experimental_strategy,
                    label=args.checkpoint_dir.split('/')[-2],
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


        # Test model
        test_metrics, test_y_c, test_y_pred_c = test_pipeline(
            test_set=test_set,
            config_json=config_json,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            fold=fold
        )
        if task_type == "regression":
            plt.title("Regression "+ args.checkpoint_dir.split('/')[-2])
            d_indices = [i for i in range(len(test_y_c))]
            plt.plot(test_y_c, d_indices, "ro", label="ground-truth")
            plt.plot(test_y_pred_c, d_indices, "bo", label="prediction")
            plt.legend()
            plt.savefig(
                fname=os.path.join(args.checkpoint_dir, f"regression_fig{fold}.png"),
                bbox_inches='tight'
            )
            plt.clf()
            plt.close()

        # Convert test metrics into a dataframe
        test_metrics_df = pd.DataFrame.from_dict(test_metrics)
        test_metrics_df.to_csv(os.path.join(args.checkpoint_dir, f"test_metrics_kf{fold}.csv"))
        print(test_metrics_df)
