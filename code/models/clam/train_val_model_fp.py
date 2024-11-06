# Imports
from __future__ import print_function
import os
import argparse
import numpy as np
import random
import datetime
import json
import shutil
import copy

# PyTorch Imports
import torch

# Project Imports
from train_val_test_utilities import train_val_pipeline
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
    parser.add_argument('--results_dir', type=str, required=True, help='The path to the results directory.')
    parser.add_argument('--dataset', type=str, required=True, choices=['TCGA-BRCA'], help='The dataset for the experiments.')
    parser.add_argument('--base_data_path', type=str, required=True, help='Base data path for TCGA-BRCA dataset.')
    parser.add_argument('--experimental_strategy', type=str, choices=['All', 'DiagnosticSlide', 'TissueSlide'], required=True, help="The experimental strategy for the TCGA-BRCA dataset.")
    parser.add_argument('--features_h5_dir', nargs='+', type=str, required=True, help="The directory of the features in .pt format for the TCGA-BRCA dataset.")
    parser.add_argument(
        '--label', 
        type=str, 
        choices=[
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
        ],
        required=False,
        help='The SSEGA pathways for the TCGA-BRCA dataset.'
    )
    parser.add_argument("--config_json", type=str, required=True, help="The path to the configuration JSON.")
    args = parser.parse_args()



    # Set seeds (for reproducibility reasons)
    set_seed(seed=args.seed)


    # Load configuration JSON
    with open(args.config_json, 'r') as j:
        config_json = json.load(j)


    # Create results directory
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    # Get timestamp and experiment directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(args.results_dir, args.label, timestamp)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    # Copy configuration JSON to the experiment directory
    _ = shutil.copyfile(
        src=args.config_json,
        dst=os.path.join(experiment_dir, 'config.json')
    )

    # Load GPU/CPU device
    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')


    # Get the encoding size for the feature vectors
    encoding_size = config_json['data']['encoding_size']

    # Get verbose
    verbose = config_json['verbose']

    # Features source
    features_ = config_json["features"]

    # Task type
    task_type = config_json["task_type"]


    # Load data
    print('Loading dataset...')
    if args.dataset == 'TCGA-BRCA':
        if task_type == "classification":
            dataset = TCGABRCA_MIL_Dataset(
                base_data_path=args.base_data_path,
                experimental_strategy=args.experimental_strategy,
                label=args.label,
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
                    label=args.label,
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
        train_set.select_fold(fold=fold)
        val_set.select_fold(fold=fold)

        # Train model
        checkpoint_fname = f"best_model_kf{fold}.pt"
        if args.label is None:
            if task_type == "clinical_subtype_classification":
                args.label = "c_subtype"
        train_val_pipeline(
            datasets=(train_set, val_set),
            config_json=config_json,
            device=device,
            experiment_dir=experiment_dir,
            checkpoint_fname=checkpoint_fname,
            wandb_project_name=config_json["hyperparameters"]["model_type"] + f"{features_}_{args.label}_{timestamp}_{fold}"
        )
