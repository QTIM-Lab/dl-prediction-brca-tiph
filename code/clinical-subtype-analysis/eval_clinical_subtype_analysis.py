# Imports
from __future__ import print_function
import os
import argparse
import numpy as np
import random
import json


# PyTorch Imports
import torch

# Project Imports
from train_val_test_utilities import compute_metrics_per_clinical_subtype



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
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='The path to the checkpoint directory.')
    args = parser.parse_args()



    # Set seeds (for reproducibility reasons)
    set_seed(seed=args.seed)


    # Load configuration JSON
    with open(os.path.join(args.checkpoint_dir, "config.json"), 'r') as j:
        config_json = json.load(j)

    # Iterate through folds
    n_folds = int(config_json["data"]["n_folds"])
    n_classes = int(config_json["data"]["n_classes"])
    for fold in range(n_folds):

        # Set seed
        set_seed(seed=args.seed)

        compute_metrics_per_clinical_subtype(checkpoint_dir=args.checkpoint_dir, n_classes=n_classes, fold=fold)