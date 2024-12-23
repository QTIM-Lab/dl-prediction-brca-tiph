# Imports
from __future__ import print_function
import os
import argparse
import numpy as np
import random
import json
import copy

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Scikit-learn Imports
from sklearn.manifold import TSNE

# Project Imports
from data_utilities import TCGABRCA_MIL_DatasetRegression
from model_utilities import AM_SB_Regression



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

    # Load the parameters from the configuration JSON
    n_classes = config_json["data"]["n_classes"]
    dropout = config_json["hyperparameters"]["dropout"]
    dropout_prob = config_json["hyperparameters"]["dropout_prob"]
    model_size = config_json["hyperparameters"]["model_size"]
    model_type = config_json["hyperparameters"]["model_type"]
    verbose = config_json["verbose"]
    num_workers = config_json["data"]["num_workers"]
    pin_memory = config_json["data"]["pin_memory"]
    encoding_size = config_json['data']['encoding_size']
    features_ = config_json["features"]
    task_type = config_json["task_type"]


    # Dictionary with model settings for the initialization of the model object
    if task_type in ("classification", "clinical_subtype_classification"):
        model_dict = {
            "dropout":dropout,
            "dropout_prob":dropout_prob,
            'n_classes':n_classes,
            "encoding_size":encoding_size
        }
    elif task_type == "regression":
        model_dict = {
            "dropout":dropout,
            "dropout_prob":dropout_prob,
            "encoding_size":encoding_size
        }
    

    # Adapted from the CLAM framework
    assert model_size is not None, "Please define a model size."
    
    model_dict.update({"size_arg": model_size})
    
    # AM-SB
    if task_type in ("classification", "clinical_subtype_classification"):
        if model_type == 'am_sb':
            pass
        elif model_type == 'am_mb':
            pass
    elif task_type == "regression":
        if model_type == 'am_sb':
            model = AM_SB_Regression(**model_dict)

    

    # Load model checkpoint
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"best_model_kf0.pt"), map_location=device))
    model.to(device)

    # Put model into evaluation 
    model.eval()


    # Experiment directory
    experiment_dir = os.path.join(args.checkpoint_dir, 'feature-analysis', 't-sne')
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)


    # Load t-SNE
    tsne = TSNE(n_components=2, random_state=args.seed)
    
    # Load data
    print('Loading dataset...')
    if args.dataset == 'TCGA-BRCA':
        dataset = TCGABRCA_MIL_DatasetRegression(
            base_data_path=args.base_data_path,
            experimental_strategy=args.experimental_strategy,
            label=args.checkpoint_dir.split('/')[-2],
            features_h5_dir=args.features_h5_dir,
            n_folds=1,
            seed=int(args.seed)
        )

        # Create the data splits from the original dataset (and the DataLoaders)
        train_set = copy.deepcopy(dataset)
        train_set.select_split(split='train')
        train_set.select_fold(fold=0)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_set = copy.deepcopy(dataset)
        val_set.select_split(split='validation')
        val_set.select_fold(fold=0)
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_set = copy.deepcopy(dataset)
        test_set.select_split(split='test')
        test_set.select_fold(fold=0)
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        pass

    
    # Create an X to train t-SNE
    X = list()
    # Go through all the dataloaders
    for data_loader in (train_loader, val_loader, test_loader):
        for _, input_data_dict in enumerate(data_loader):
            features, ssgsea_scores, ssgsea_scores_bin = input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device)
            output_dict = model(features)
            features_att = output_dict['features']
            # y_pred = torch.where(logits > 0, 1.0, 0.0)
            # y_pred_proba = F.sigmoid(logits)
            # test_y_pred_c.extend(list(logits.squeeze(0).cpu().detach().numpy()))
            # test_y_pred.extend(list(y_pred.squeeze(0).cpu().detach().numpy()))
            # test_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))
            # test_y_c.extend(list(ssgsea_scores.cpu().detach().numpy()))
            # test_y_pred_proba.extend(list(y_pred_proba.squeeze(0).cpu().detach().numpy()))
            X.extend(features_att.squeeze().cpu().detach().numpy())

    # Train t-sne
    X = np.array(X)
    print("X.shape ", X.shape)
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE KL Divergence: {tsne.kl_divergence_}")