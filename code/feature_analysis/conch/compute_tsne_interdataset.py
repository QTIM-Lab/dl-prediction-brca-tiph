# Imports
from __future__ import print_function
import os
import argparse
import numpy as np
import random
import json
import copy
import matplotlib.pyplot as plt

# PyTorch Imports
import torch
from torch.utils.data import DataLoader

# Scikit-learn Imports
from sklearn.manifold import TSNE

# Project Imports
from data_utilities import TCGABRCA_MIL_DatasetRegression, OhioStatePathDataset, MGHPathDataset
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
    experiment_dir = os.path.join(args.checkpoint_dir, 'feature-analysis', 't-sne-interdataset')
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)


    # Load t-SNE
    tsne = TSNE(n_components=2, random_state=args.seed)
    
    # Load data
    print('Loading dataset...')
    label = args.checkpoint_dir.split('/')[-2]

    # TCGA-BRCA
    dataset = TCGABRCA_MIL_DatasetRegression(
        base_data_path='/autofs/space/crater_001/datasets/public/TCGA-BRCA',
        experimental_strategy='All',
        label=label,
        features_h5_dir=['/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features', '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features'],
        n_folds=1,
        seed=int(args.seed)
    )
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


    # Ohio State
    o_dataset = OhioStatePathDataset(
        base_data_path='/autofs/space/crater_001/datasets/private/breast_path/ohio_state_breast_ICI/',
        features_h5_dir='/autofs/space/crater_001/datasets/private/breast_path/ohio_state_breast_ICI/SegmentationCLAM/features/CONCH/',
        seed=int(args.seed)
    )
    o_test_loader = DataLoader(
        dataset=o_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # MGH
    m_dataset = MGHPathDataset(
        base_data_path='/autofs/space/crater_001/datasets/private/breast_path/MGH_breast/',
        features_h5_dir='/autofs/space/crater_001/datasets/private/breast_path/MGH_breast/CLAM/tcga/features/CONCH/',
        seed=int(args.seed)
    )
    m_test_loader = DataLoader(
        dataset=m_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


    
    # Create an X to train t-SNE
    X = list()
    y = list()
    # Go through all the dataloaders
    with torch.no_grad():
        for data_loader in (train_loader, val_loader, test_loader):
            for _, input_data_dict in enumerate(data_loader):
                features = input_data_dict["features"].to(device)
                output_dict = model(features)
                features_att = output_dict['features']
                X.extend(features_att.cpu().detach().numpy())
                y.append(0)
        
        for _, input_data_dict in enumerate(o_test_loader):
            features = input_data_dict["features"].to(device)
            output_dict = model(features)
            features_att = output_dict['features']
            X.extend(features_att.cpu().detach().numpy())
            y.append(1)
        
        for _, input_data_dict in enumerate(m_test_loader):
            features = input_data_dict["features"].to(device)
            output_dict = model(features)
            features_att = output_dict['features']
            X.extend(features_att.cpu().detach().numpy())
            y.append(2)

    

    # Dataset dictionary
    dataset_dict = {
        0:"TCGA-BRCA",
        1:"Ohio State",
        2:"MGH"
    }
    # Train t-sne
    X = np.array(X)
    y = np.array(y)
    label_ = [dataset_dict[c] for c in y]
    
    # t-SNE
    X_tsne = tsne.fit_transform(X)
    clset = set(zip(y, label_))
    ax = plt.gca()
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker="o")[0] for c,l in clset]
    labels = [l for c,l in clset]
    ax.legend(handles, labels, loc='best')
    plt.title(f't-SNE: {label}')
    ax.set_xlabel('1st t-SNE Component')
    ax.set_ylabel('2nd t-SNE Component')
    plt.savefig(
        fname=os.path.join(experiment_dir, 'tsne_id_plot.png'),
        bbox_inches='tight'
    )
    plt.savefig(
        fname=os.path.join('results', f'tsne_id_plot_{label}.png'),
        bbox_inches='tight'
    )
    plt.clf()
    plt.close()