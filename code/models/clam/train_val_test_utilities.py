# Imports
import os
import numpy as np

# PyTorch Imports
import torch
import torch.nn as nn
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

# Sklearn Imports
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score

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



# Function: Train-Validation Model Pipeline
def train_val_pipeline(datasets, config_json, device, experiment_dir, checkpoint_fname, wandb_project_name, early_stopping=False, patience=20, stop_epoch=50, verbose=True):

    # Load the parameters from the configuration JSON
    n_classes = config_json["data"]["n_classes"]
    dropout = config_json["hyperparameters"]["dropout"]
    dropout_prob = config_json["hyperparameters"]["dropout_prob"]
    model_size = config_json["hyperparameters"]["model_size"]
    model_type = config_json["hyperparameters"]["model_type"]
    verbose = config_json["verbose"]
    optimizer = config_json["hyperparameters"]["optimizer"]
    lr = config_json["hyperparameters"]["lr"]
    weight_decay = config_json["hyperparameters"]["weight_decay"]
    num_workers = config_json["data"]["num_workers"]
    pin_memory = config_json["data"]["pin_memory"]
    epochs = config_json["hyperparameters"]["epochs"]
    early_stopping = config_json["hyperparameters"]["early_stopping"]
    loss = config_json["hyperparameters"]["loss"]
    encoding_size = config_json['data']['encoding_size']
    features_ = config_json["features"]
    task_type = config_json["task_type"]


    # Build a configuration dictionary for WandB
    wandb_project_config = {
        "n_classes":n_classes,
        "dropout":dropout,
        "dropout_prob":dropout_prob,
        "model_size":model_size,
        "model_type":model_type,
        "verbose":verbose,
        "optimizer":optimizer,
        "lr":lr,
        "weight_decay":weight_decay,
        "num_workers":num_workers,
        "pin_memory":pin_memory,
        "epochs":epochs,
        "early_stopping":early_stopping,
        "loss":loss,
        "encoding_size":encoding_size,
        "features":features_,
        "task_type":task_type
    }

    # Initialize WandB
    wandb_run = wandb.init(
        project="dl-prediction-brca-tiph-ext",
        name=wandb_project_name,
        config=wandb_project_config
    )
    assert wandb_run is wandb.run



    # Get data splits
    train_set, val_set = datasets


    # Get loss function
    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "regression":
        loss_fn = nn.MSELoss()


    # Dictionary with model settings for the initialization of the model object
    if task_type == "classification":
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
    if task_type == "classification":
        if model_type == 'am_sb':
            model = AM_SB(**model_dict)
        elif model_type == 'am_mb':
            model = AM_MB(**model_dict)
    elif task_type == "regression":
        if model_type == 'am_sb':
            model = AM_SB_Regression(**model_dict)
    

    # Move into model into device
    model.to(device=device)

    if verbose:
        summary(model)

    
    # Get and load the optimizer
    optimizer = get_optim(
        model=model,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay
    )
    

    # Create DataLoaders 
    # FIXME: The code is currently optimized for batch_size == 1)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


    # Tracking parameters
    tracking_params = {"min_val_loss":np.inf}
    if early_stopping:
        tracking_params["patience"] = patience
        tracking_params["stop_epoch"] = stop_epoch
        tracking_params["early_stopping_counter"] = 0
        tracking_params["early_stop"] = False


    # Training Pipeline
    for epoch in range(epochs):
        train_loop_clam(
            epoch=epoch, 
            model=model, 
            loader=train_loader, 
            optimizer=optimizer, 
            n_classes=n_classes,
            task_type=task_type,
            loss_fn=loss_fn,
            device=device,
            wandb_run=wandb_run
        )
        validate_loop_clam(
            model=model, 
            loader=val_loader, 
            n_classes=n_classes, 
            task_type=task_type,
            tracking_params=tracking_params, 
            loss_fn=loss_fn, 
            experiment_dir=experiment_dir,
            checkpoint_fname=checkpoint_fname,
            device=device,
            wandb_run=wandb_run
        )

        # Stop training according to the early stopping parameters
        if early_stopping:
            if tracking_params["early_stop"]: 
                break

    return



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


    # Dictionary with model settings for the initialization of the model object
    model_dict = {
        "dropout":dropout,
        "dropout_prob":dropout_prob,
        'n_classes':n_classes,
        "encoding_size":encoding_size
    }
    

    # Adapted from the CLAM framework
    assert model_size is not None, "Please define a model size."
    
    model_dict.update({"size_arg": model_size})

    if model_type == 'am_sb':
        model = AM_SB(**model_dict)
    elif model_type == 'am_mb':
        model = AM_MB(**model_dict)

    # Move into model into device
    model.to(device=device)

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
    

    # Put model into evaluation 
    model.eval()

    # Data dictionary to log step accuracies
    data = [{"count": 0, "correct": 0} for _ in range(n_classes)]
    
    # Initialize variables to track values
    test_error = 0.
    test_y_pred = list()
    test_y_pred_prob = list()
    test_y = list()

    # Create a dictionary for test metrics
    test_metrics = dict()


    # Go through data batches and get metric values
    with torch.no_grad():
        for _, input_data_dict in enumerate(test_loader):
            features, ssgea_scores = input_data_dict['features'].to(device), input_data_dict['ssgsea_scores'].to(device)
            output_dict = model(features)
            logits, y_pred_prob, y_pred = output_dict['logits'], output_dict['y_proba'], output_dict['y_pred'], 
            
            # Log step counts and correct counts
            data[int(ssgea_scores)]["count"] += 1
            data[int(ssgea_scores)]["correct"] += (int(y_pred) == int(ssgea_scores))

            # Append predictions and labels into the corresponding lists
            test_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            test_y.extend(list(ssgea_scores.cpu().detach().numpy()))

            # Updated list of y_pred probabilities
            test_y_pred_prob.append(y_pred_prob.cpu().numpy())

            # Compute error and update it
            error = compute_prediction_error(y_pred=y_pred, y=ssgea_scores)
            test_error += error


    # Updated final validation error
    test_error /= len(test_loader)

    # Append test error to the metrics
    test_metrics["test_error"] = [test_error]


    # Compute Validation ROC AUC
    if n_classes == 2:

        # Convert lists to NumPy arrays and fix their shape(s)
        test_y = np.array(test_y)
        test_y_pred_prob = np.squeeze(np.array(test_y_pred_prob), axis=1)

        # Compute the ROC AUC Score
        test_auc = roc_auc_score(test_y, test_y_pred_prob[:, 1])
        test_aucs = []
    
    else:
        test_aucs = []
        binary_labels = label_binarize(test_y, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in test_y:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], test_y_pred_prob[:, class_idx])
                test_aucs.append(auc(fpr, tpr))
            else:
                test_aucs.append(float('nan'))

        test_auc = np.nanmean(np.array(test_aucs))
    

    # Append test AUC to the test metrics
    test_metrics["test_auc"] = [test_auc]

    # Compute overall accuracy
    test_acc = accuracy_score(y_true=test_y, y_pred=test_y_pred)
    test_metrics["test_acc"] = [test_acc]


    # Compute class accuracies
    acc_dict, _, _ = compute_class_acc(data=data, n_classes=n_classes)
    for c in range(n_classes):
        test_metrics[f"test_class_{c}_acc"] = [acc_dict[c]]


    return test_metrics



# Function: Train Loop for CLAM
def train_loop_clam(epoch, model, loader, optimizer, n_classes, task_type, loss_fn, device, wandb_run):

    # Put model into training mode
    model.train()

    # Initialize variables 
    train_loss = 0.
    train_y_pred = list()
    train_y_pred_proba = list()
    train_y = list()

    # Get batch of data
    for _, input_data_dict in enumerate(loader):

        if task_type == "classification":
            features, ssgsea_scores = input_data_dict['features'].to(device), input_data_dict['ssgsea_scores'].to(device)
            output_dict = model(features)
            logits, y_pred = output_dict['logits'], output_dict['y_pred']
            train_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            train_y.extend(list(ssgsea_scores.cpu().detach().numpy()))

        elif task_type == "regression":
            features, ssgsea_scores, ssgsea_scores_bin = input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device)
            output_dict = model(features)
            logits = output_dict['logits']
            y_pred = torch.where(logits > 0, 1.0, 0.0)
            train_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            train_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))

        # Compute train loss
        loss = loss_fn(logits, ssgsea_scores)

        # Get loss values and update records
        loss_value = loss.item()
        train_loss += loss_value

        # Log batch metrics to WandB
        wandb_run.log({"train_batch_loss":loss_value})
        
        # Backpropagation 
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()



    # Calculate loss and error for epoch
    train_loss /= len(loader)


    # Compute metrics
    train_y_pred = torch.from_numpy(np.array(train_y_pred))
    train_y = torch.from_numpy(np.array(train_y))

    if n_classes == 2:
        acc = accuracy(
            preds=y_pred,
            target=train_y,
            task='binary'
        )

        f1 = f1_score(
            preds=y_pred,
            target=train_y,
            task='binary'
        )

        rec = recall(
            preds=y_pred,
            target=train_y,
            task='binary'
        )

        prec = precision(
            preds=y_pred,
            target=train_y,
            task='binary'
        )


        # TODO: Review this in the meantime, it applies to the classification but not to regression
        # auc = auroc(
        #     preds=y_pred_proba,
        #     target=y,
        #     task='binary'
        # )

    else:
        pass

    
    # Log metrics into W&B
    wandb_run.log(
        {
            "train_epoch":epoch,
            "train_loss":train_loss,
            "train_acc":acc,
            "train_f1":f1,
            "train_rec":rec,
            "train_prec":prec
            # "train_auc":auc
        }
    )

    return



# Function: Validation Loop for CLAM
def validate_loop_clam(model, loader, n_classes, task_type, tracking_params, loss_fn, experiment_dir, checkpoint_fname, device, wandb_run):

    # Put model into evaluation 
    model.eval()
    
    # Initialize variables to track values
    val_loss = 0.
    val_y_pred = list()
    val_y_pred_proba = list()
    val_y = list()


    # Go through data batches and get metric values
    with torch.no_grad():
        for _, input_data_dict in enumerate(loader):
            if task_type == "classification":
                features, ssgsea_scores = input_data_dict['features'].to(device), input_data_dict['ssgsea_scores'].to(device)
                output_dict = model(features)
                logits, y_pred = output_dict['logits'], output_dict['y_pred']
                val_y_pred.extend(list(y_pred.cpu().detach().numpy()))
                val_y.extend(list(ssgsea_scores.cpu().detach().numpy()))
                # val_y_pred_proba.extend(list(y_pred_proba.cpu().detach().numpy()))

            elif task_type == "regression":
                features, ssgsea_scores, ssgsea_scores_bin = input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device)
                output_dict = model(features)
                logits = output_dict['logits']
                y_pred = torch.where(logits > 0, 1.0, 0.0)
                val_y_pred.extend(list(y_pred.cpu().detach().numpy()))
                val_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))


            # Compute validation loss
            loss = loss_fn(logits, ssgsea_scores)
            loss_value = loss.item()
            val_loss += loss_value

            # Log batch metrics to WandB
            wandb_run.log({"val_batch_loss":loss_value})


    # Updated final validation loss
    val_loss /= len(loader)

    # Compute metrics
    val_y_pred = torch.from_numpy(np.array(val_y_pred))
    val_y = torch.from_numpy(np.array(val_y))

    if n_classes == 2:
        acc = accuracy(
            preds=val_y_pred,
            target=val_y,
            task='binary'
        )

        f1 = f1_score(
            preds=val_y_pred,
            target=val_y,
            task='binary'
        )

        rec = recall(
            preds=val_y_pred,
            target=val_y,
            task='binary'
        )

        prec = precision(
            preds=val_y_pred,
            target=val_y,
            task='binary'
        )


        # TODO: Review this in the meantime, it applies to the classification but not to regression
        # auc = auroc(
        #     preds=val_y_pred_proba,
        #     target=val_y,
        #     task='binary'
        # )

    else:
        pass

    
    # Log metrics into W&B
    wandb_run.log(
        {
            "val_loss":val_loss,
            "val_acc":acc,
            "val_f1":f1,
            "val_rec":rec,
            "val_prec":prec
            # "val_auc":auc
        }
    )


    # Save checkpoints based on tracking_params parameters
    if tracking_params is not None:
        
        assert experiment_dir
        
        if val_loss < tracking_params["min_val_loss"]:
            tracking_params["min_val_loss"] = val_loss
            torch.save(model.state_dict(), os.path.join(experiment_dir, checkpoint_fname))
            if "early_stop" in tracking_params.keys():
                tracking_params["counter"] = 0
        else:
            if "early_stop" in tracking_params.keys():
                tracking_params["counter"] += 1
                if tracking_params["counter"] >= tracking_params["patience"]:
                    tracking_params["early_stop"] = True

    return
