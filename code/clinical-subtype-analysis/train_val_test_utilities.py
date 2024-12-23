# Imports
import os
import numpy as np
import pandas as pd

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

# TorchMetrics Imports
from torchmetrics.functional.classification import (
    accuracy,
    f1_score,
    recall,
    precision,
    auroc
)
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_squared_error,
    concordance_corrcoef,
    kendall_rank_corrcoef,
    pearson_corrcoef,
    r2_score,
    relative_squared_error,
    spearman_corrcoef
)

# Sklearn Imports
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

# Project Imports
from model_utilities import AM_SB, AM_MB, AM_SB_Regression

# WandB Imports
import wandb



# Function: Get optimizer
def get_optim(model, optimizer, lr, weight_decay):
    if optimizer == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer



# Function: Test Model Pipeline
def test_pipeline(test_set, config_json, device, checkpoint_dir, fold):

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
            model = AM_SB(**model_dict)
        elif model_type == 'am_mb':
            model = AM_MB(**model_dict)
    elif task_type == "regression":
        if model_type == 'am_sb':
            model = AM_SB_Regression(**model_dict)

    if verbose:
        print(f"Using features: {features_}")
        summary(model)


    # Create DataLoaders 
    # FIXME: The code is currently optimized for batch_size == 1)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Load model checkpoint
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best_model_kf{fold}.pt"), map_location=device))
    model.to(device)

    # Put model into evaluation 
    model.eval()

    # Initialize variables to track values
    test_y_pred = list()
    test_y_pred_proba = list()
    test_y = list()

    # Create a dictionary for test metrics
    test_inference_info = dict()


    # Initialize variables 
    test_y_pred = list()
    test_y_pred_proba = list()
    test_y_pred_c = list()
    test_y = list()
    test_y_c = list()
    test_y_cs = list()
    case_ids = list()
    svs_paths = list()

    # Get batch of data
    for _, input_data_dict in enumerate(test_loader):

        features, ssgsea_scores, ssgsea_scores_bin, c_subtype, case_id, svs_path, = input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device), input_data_dict["c_subtype"], input_data_dict["case_id"], input_data_dict["svs_path"]
        output_dict = model(features)
        logits = output_dict['logits']
        y_pred = torch.where(logits > 0, 1.0, 0.0)
        y_pred_proba = F.sigmoid(logits)
        test_y_pred_c.extend(list(logits.squeeze(0).cpu().detach().numpy()))
        test_y_pred.extend(list(y_pred.squeeze(0).cpu().detach().numpy()))
        test_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))
        test_y_c.extend(list(ssgsea_scores.cpu().detach().numpy()))
        test_y_pred_proba.extend(list(y_pred_proba.squeeze(0).cpu().detach().numpy()))
        test_y_cs.extend(c_subtype)
        case_ids.extend(case_id)
        svs_paths.extend(svs_path)


    # Test inference information
    test_inference_info["case_id"] = case_ids
    test_inference_info["svs_path"] = svs_paths
    test_inference_info["ssgsea_c"] = test_y_c
    test_inference_info["ssgsea_b"] = test_y
    test_inference_info["ssgsea_c_pred"] = test_y_pred_c
    test_inference_info["ssgsea_b_pred"] = test_y_pred
    test_inference_info["ssgsea_b_pred_proba"] = test_y_pred_proba
    test_inference_info["c_subtype"] = test_y_cs

    return test_inference_info



# Function: Get metrics per clinical subtype and split
def compute_metrics_per_clinical_subtype(checkpoint_dir, n_classes=2, fold=0):

    # Get label
    label = checkpoint_dir.split('/')[-2]

    # Go through the possible evaluation names
    for eval_name in ('train', 'val', 'test'):
        csv_fpath = os.path.join(checkpoint_dir, f"{eval_name}_inference_info_kf{fold}.csv")
        info_df = pd.read_csv(csv_fpath)
        eval_metrics = dict()

        # Go through the possible clinical subtypes
        for c_subtype, c_subtype_name in zip(("HER2+/HR+", "HER2+/HR-", "HER2-/HR+", "HER2-/HR-"),("her2pos_hrpos", "her2pos_hrneg", "her2neg_hrpos", "her2neg_hrneg")):
            info_df_ = info_df.copy()[info_df["c_subtype"]==c_subtype]

            # Get needed values
            y = torch.from_numpy(info_df_["ssgsea_b"].values)
            y_c = torch.from_numpy(info_df_["ssgsea_c"].values)
            y_pred = torch.from_numpy(info_df_["ssgsea_b_pred"].values)
            y_pred_proba = torch.from_numpy(info_df_["ssgsea_b_pred_proba"].values)
            y_pred_c = torch.from_numpy(info_df_["ssgsea_c_pred"].values)

            # Compute metrics
            if n_classes == 2:
                if len(test_y_pred.shape) == 2:
                    test_y_pred = test_y_pred.squeeze()
                
                acc = accuracy(
                    preds=test_y_pred,
                    target=y,
                    task='binary'
                )

                f1 = f1_score(
                    preds=test_y_pred,
                    target=y,
                    task='binary'
                )

                rec = recall(
                    preds=test_y_pred,
                    target=y,
                    task='binary'
                )

                prec = precision(
                    preds=test_y_pred,
                    target=y,
                    task='binary'
                )

                auc = auroc(
                    preds=y_pred_proba,
                    target=y,
                    task='binary'
                )

            else:
                acc = accuracy(
                    preds=test_y_pred,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

                f1 = f1_score(
                    preds=test_y_pred,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

                rec = recall(
                    preds=test_y_pred,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

                prec = precision(
                    preds=test_y_pred,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

                auc = auroc(
                    preds=y_pred_proba,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )
            
            # Regression Metrics
            mae = mean_absolute_error(
                preds=y_pred_c,
                target=y_c
            )
            
            mse = mean_squared_error(
                preds=y_pred_c,
                target=y_c
            )

            ccc = concordance_corrcoef(
                preds=y_pred_c,
                target=y_c
            )

            krcc = kendall_rank_corrcoef(
                preds=y_pred_c,
                target=y_c
            )

            pcc = pearson_corrcoef(
                preds=y_pred_c,
                target=y_c
            )

            r2s = r2_score(
                preds=y_pred_c,
                target=y_c
            )

            rse = relative_squared_error(
                preds=y_pred_c,
                target=y_c
            )

            scc = spearman_corrcoef(
                preds=y_pred_c,
                target=y_c
            )

            # Append test AUC to the test metrics
            eval_metrics["acc"] = [acc.item()]
            eval_metrics["f1"] = [f1.item()]
            eval_metrics["rec"] = [rec.item()]
            eval_metrics["prec"] = [prec.item()]
            eval_metrics["auc"] = [auc.item()]
            eval_metrics["mae"] = [mae.item()]
            eval_metrics["mse"] = [mse.item()]
            eval_metrics["ccc"] = [ccc.item()]
            eval_metrics["krcc"] = [krcc.item()]
            eval_metrics["pcc"] = [pcc.item()]
            eval_metrics["r2s"] = [r2s.item()]
            eval_metrics["rse"] = [rse.item()]
            eval_metrics["scc"] = [scc.item()]

            eval_metrics_info_df = pd.DataFrame.from_dict(eval_metrics)
            eval_metrics_info_df.to_csv(os.path.join(checkpoint_dir, f"{eval_name}_{c_subtype_name}_eval_metrics_info_kf{fold}.csv"))
            os.makedirs(os.path.join('results', label), exist_ok=True)
            eval_metrics_info_df.to_csv(os.path.join('results', label, f"{eval_name}_{c_subtype_name}_eval_metrics_info_kf{fold}.csv"))