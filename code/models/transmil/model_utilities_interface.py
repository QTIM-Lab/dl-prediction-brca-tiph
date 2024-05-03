# Imports
import os
import pandas as pd

# PyTorch & PyTorch Ligthning Imports
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

# Project Imports
from model_utilities_transmil import TransMIL
from optimizer_utilities import create_optimizer
from loss_utilities import create_loss



# Function: Log step statistics
def compute_and_log_step_stats(y_pred, y, data_dict):

    y_pred_ = int(y_pred)
    y_ = int(y)
    data_dict[y_]["count"] += 1
    data_dict[y_]["correct"] += (y_pred_ == y_)

    return



# Function: Compute class accuracies in each step
def compute_step_acc(data_dict, n_classes):

    # Create dictionaries for the metrics
    acc_dict, correct_dict, count_dict = dict(), dict(), dict()

    # Go through the classes
    for c in range(n_classes):
        count = data_dict[c]["count"]
        correct = data_dict[c]["correct"]
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        # print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        acc_dict[c] = acc
        correct_dict[c] = correct
        count_dict[c] = count

    return acc_dict, correct_dict, count_dict



# Class: ModelInterface
class  ModelInterface(pl.LightningModule):

    # Method: __init__
    def __init__(self, model, loss, optimizer, **kwargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()

        # Number of classes
        self.n_classes = model['n_classes']

        # Feature encoding size and its source
        self.encoding_size = kwargs['encoding_size']
        self.features = kwargs['features']

        # Load model
        self.load_model()

        # Create train and validation loss functions
        self.loss = create_loss(loss['base_loss'])
        self.val_loss = nn.CrossEntropyLoss()

        # Create optimizer
        self.optimizer = optimizer

        # Create log path to save the logs of the experiment
        self.log_path = kwargs['log']

        #  Data dictionary to log step accuracies
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

        #  Metrics, byt TorchMetrics
        self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average='macro', task='multiclass'),
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(num_classes=self.n_classes, average='micro', task='multiclass'),
                torchmetrics.CohenKappa(num_classes=self.n_classes, task='multiclass'),
                torchmetrics.F1Score(num_classes=self.n_classes, average='macro', task='multiclass'),
                torchmetrics.Recall(average='macro', num_classes=self.n_classes, task='multiclass'),
                torchmetrics.Precision(average='macro', num_classes=self.n_classes, task='multiclass'),
                torchmetrics.Specificity(average='macro', num_classes=self.n_classes, task='multiclass')
            ]
        )

        # Create variables for train, validation and test metrics
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        return


    # Function: Progress bar dictionary
    def get_progress_bar_dict(self):

        # Don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


    # Method: Training Step
    def training_step(self, batch, batch_idx):

        # Get the features
        features = batch['features']

        # Get the ssGSEA Scores
        ssgsea_scores = batch['ssgsea_scores']

        # Build data variables to optimize the model
        data, label = features, ssgsea_scores
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # Compute loss
        loss = self.loss(logits, label)

        # Compute and log step statistics
        compute_and_log_step_stats(
            y_pred=Y_hat,
            y=label,
            data_dict=self.data
        )

        # Create a dictionary for training step outputs
        training_step_outputs = {
            'loss': loss, 
            'logits' : logits, 
            'Y_prob' : Y_prob, 
            'Y_hat' : Y_hat, 
            'label' : label
        }

        return training_step_outputs


    # Method: training_epoch_end
    def training_epoch_end(self, training_step_outputs):

        # Get the training step outputs
        loss = torch.tensor([x['loss'] for x in training_step_outputs]).mean()
        probs = torch.cat([x['Y_prob'] for x in training_step_outputs], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in training_step_outputs])
        target = torch.stack([x['label'] for x in training_step_outputs], dim=0)


        # Compute loss on train
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, logger=True)

        # Compute AUROC on train
        self.log('train_auroc',  self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)

        # Log metrics to WandB
        self.log_dict(self.train_metrics(max_probs.squeeze() , target.squeeze()), on_epoch=True, logger=True)


        # Compute step accuracies
        acc_dict, correct_dict, count_dict = compute_step_acc(
            data_dict=self.data,
            n_classes=self.n_classes
        )
        for c in range(self.n_classes):
            self.log(f'train_acc_class_{c}', acc_dict[c], prog_bar=True, on_epoch=True, logger=True)
            self.log(f'train_correct_class_{c}', correct_dict[c], prog_bar=True, on_epoch=True, logger=True)
            self.log(f'train_count_class_{c}', count_dict[c], prog_bar=True, on_epoch=True, logger=True)


        # Reset data dictonary
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

        return


    # Method: validation_step
    def validation_step(self, batch, batch_idx):

        # Get the features
        features = batch['features']
        
        # Get the ssGSEA scores
        ssgsea_scores = batch['ssgsea_scores']

        # Build data variables
        data, label = features, ssgsea_scores
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # Compute and log step statistics
        compute_and_log_step_stats(
            y_pred=Y_hat,
            y=label,
            data_dict=self.data
        )

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    # Method: validation_epoch_end
    def validation_epoch_end(self, val_step_outputs):

        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)

        # Compute loss on validation
        self.log('val_loss', self.val_loss(logits, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)

        # Compute AUC
        self.log('val_auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)

        # Log metrics to WandB
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()), on_epoch = True, logger = True)

        # Compute step accuracies
        acc_dict, correct_dict, count_dict = compute_step_acc(
            data_dict=self.data,
            n_classes=self.n_classes
        )
        for c in range(self.n_classes):
            self.log(f'val_acc_class_{c}', acc_dict[c], prog_bar=True, on_epoch=True, logger=True)
            self.log(f'val_correct_class_{c}', correct_dict[c], prog_bar=True, on_epoch=True, logger=True)
            self.log(f'val_count_class_{c}', count_dict[c], prog_bar=True, on_epoch=True, logger=True)

        # Reset data dictionary
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

        return


    # Method: configure_optimizers
    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]


    # Method: test_step
    def test_step(self, batch, batch_idx):

        # Get the features
        features = batch['features']
        
        # Get the ssGSEA scores
        ssgsea_scores = batch['ssgsea_scores']

        # Prepare the remaining data variables
        data, label = features, ssgsea_scores
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # Compute and log step statistics
        compute_and_log_step_stats(
            y_pred=Y_hat,
            y=label,
            data_dict=self.data
        )

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    # Method: test_epoch_end
    def test_epoch_end(self, test_step_outputs):

        # Create a dictionary for test metrics
        test_metrics = dict()

        # Get test step outputs
        probs = torch.cat([x['Y_prob'] for x in test_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in test_step_outputs])
        target = torch.stack([x['label'] for x in test_step_outputs], dim = 0)

        # Compute performance metrics
        test_auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())

        # Add these metrics to the dictionary
        test_metrics['test_auc'] = [test_auc.cpu().numpy().item()]
        for keys, values in metrics.items():
            test_metrics[keys] = [values.cpu().numpy().item()]

        # Compute step accuracies
        acc_dict, _, _ = compute_step_acc(
            data_dict=self.data,
            n_classes=self.n_classes
        )
        for c in range(self.n_classes):
            test_metrics[f'test_acc_class_{c}'] = [acc_dict[c]]

        # Reset data dictionary
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

        # Save a .CSV file with the results
        result = pd.DataFrame.from_dict(test_metrics)
        result.to_csv(os.path.join(self.log_path, 'result.csv'))

        return test_metrics


    # Method: load_model
    def load_model(self):
        
        # Load TransMIL model by name
        if self.hparams['model']['name'] in ('TransMIL'):
            self.model = TransMIL(
                n_classes=self.n_classes,
                encoding_size=self.encoding_size,
            )
        
        return
