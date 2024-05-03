# Imports
import os
import argparse
import datetime
import json
import shutil

# PyTorch & PyTorch Ligthning Imports
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Project Imports
from data_utilities_interface import DataInterface
from model_utilities_interface import ModelInterface



if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(description='TransMIL: Model Training.')
    parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU we will use to run the program.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--results_dir', type=str, help='The path to the results directory.')
    parser.add_argument('--dataset', type=str, required=True, choices=['TCGA-BRCA'], help='The dataset for the experiments.')
    parser.add_argument('--base_data_path', type=str, required=True, help='Base data path for TCGA-BRCA dataset.')
    parser.add_argument('--experimental_strategy', type=str, choices=['All', 'DiagnosticSlide', 'TissueSlide'], required=True, help="The experimental strategy for the TCGA-BRCA dataset.")
    parser.add_argument('--features_h5_dir', nargs='+', type=str, required=False, help="The directory of the features in .pt format for the TCGA-BRCA dataset.")
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
        help='The SSEGA pathways for the TCGA-BRCA dataset.'
    )
    parser.add_argument("--config_json", type=str, help="The path to the configuration JSON.")
    parser.add_argument('--checkpoint_dir', type=str, help='The path to the checkpoint directory.')
    parser.add_argument("--train_or_test", type=str, default='train', choices=['train', 'test'], help="Train or Test Mode.")
    args = parser.parse_args()



    # Train
    if args.train_or_test == 'train':

        assert args.results_dir is not None
        assert args.config_json is not None
        assert args.label is not None

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
    
    # Test
    else:

        assert args.checkpoint_dir is not None

        # Get label from checkpoint directory
        args.label = args.checkpoint_dir.split('/')[-2]

        # Load configuration JSON
        with open(os.path.join(args.checkpoint_dir, "config.json"), 'r') as j:
            config_json = json.load(j)



    # Load GPU/CPU device
    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')


    # Get verbose
    verbose = config_json['general']['verbose']


    # Load data
    print('Loading dataset...')
    if args.dataset == 'TCGA-BRCA':
            dataset_args = {
                "base_data_path":args.base_data_path,
                "experimental_strategy":args.experimental_strategy,
                "setting":args.setting,
                "label":args.label,
                "label_thresh_metric":config_json["data"]["label_thresh_metric"],
                "features_pt_dir":args.features_pt_dir,
                "n_folds":int(config_json["data"]["n_folds"]),
                "seed":int(args.seed),
            }



    # Iterate through folds
    n_folds = int(config_json["data"]["n_folds"])
    for fold in range(n_folds):

        # Initialize seed
        pl.seed_everything(int(args.seed))


        # Train
        if args.train_or_test == 'train':
        
            # Load WandB logger
            wandb_project_config = {

                # General
                "comment":config_json['general']['comment'],
                "fp16":config_json['general']['fp16'],
                "amp_level":config_json['general']['amp_level'],
                "precision":config_json['general']['precision'],
                "multi_gpu_mode":config_json['general']['multi_gpu_mode'],
                "max_epochs":config_json['general']['max_epochs'],
                "grad_acc":config_json['general']['grad_acc'],
                "frozen_bn":config_json['general']['frozen_bn'],
                "early_stopping":config_json['general']['early_stopping'],
                "patience":config_json['general']['patience'],
                "verbose":config_json['general']['verbose'],
                "encoding_size":config_json['general']['encoding_size'],
                "features":config_json['general']['features'],

                # Data
                "dataset_name":config_json['data']['dataset_name'],
                "n_folds":config_json['data']['n_folds'],
                "batch_size":config_json['data']['batch_size'],
                "num_workers":config_json['data']['num_workers'],
                "pin_memory":config_json['data']['pin_memory'],
                "name":config_json['model']['name'],
                "n_classes":config_json['model']['n_classes'],

                # Hyperparameters: optimizer
                "opt":config_json['hyperparameters']['optimizer']['opt'],
                "lr":config_json['hyperparameters']['optimizer']['lr'],
                "opt_eps":config_json['hyperparameters']['optimizer']['opt_eps'],
                "opt_betas":config_json['hyperparameters']['optimizer']['opt_betas'],
                "momentum":config_json['hyperparameters']['optimizer']['momentum'],
                "weight_decay":config_json['hyperparameters']['optimizer']['weight_decay'],

                # Hyperparameters: loss
                "base_loss":config_json['hyperparameters']['loss']['base_loss']
            }
            wandblogger = WandbLogger(
                project="mmodal-xai-brca-path",
                name=config_json['model']['name'].lower() + "_" + config_json['general']['features'] + f"_{args.label}_{timestamp}_{fold}",
                config=wandb_project_config
            )


            # Load callbacks
            callbacks = list()

            # Early Stopping
            if config_json['general']['early_stopping']:
                early_stopping_callback = EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.00,
                    patience=config_json['general']['patience'],
                    verbose=verbose,
                    mode='min'
                )
                callbacks.append(early_stopping_callback)

            # Model Checkpoint
            if args.train_or_test == 'train' :
                callbacks.append(
                    ModelCheckpoint(
                        monitor='val_loss',
                        dirpath=experiment_dir,
                        filename=f'best_model_kf{fold}',
                        verbose=verbose,
                        save_last=True,
                        save_top_k=1,
                        mode='min',
                        save_weights_only=True
                    )
                )



        # Define Data Configuration and Load DataModule
        DataInterface_dict = {
            'batch_size':config_json['data']['batch_size'],
            'num_workers':config_json['data']['num_workers'],
            'pin_memory':config_json['data']['pin_memory'],
            'dataset_name':config_json['data']['dataset_name'],
            'fold':fold,
            'dataset_args':dataset_args
        }
        datamodule = DataInterface(**DataInterface_dict)


        # Define Model Configuration and Load Model
        ModelInterface_dict = {
            'model': config_json['model'],
            'loss': config_json['hyperparameters']['loss'],
            'optimizer': config_json['hyperparameters']['optimizer'],
            'data': config_json['data'],
            'log': experiment_dir if args.train_or_test == 'train' else args.checkpoint_dir,
            'encoding_size':config_json['general']['encoding_size'],
            'features':config_json['general']['features']
        }
        model = ModelInterface(**ModelInterface_dict)


        # Instantiate Trainer
        trainer = Trainer(
            gpus=str(args.gpu_id),
            num_sanity_val_steps=0, 
            logger=wandblogger if args.train_or_test == 'train' else None,
            callbacks=callbacks if args.train_or_test == 'train' else None,
            max_epochs=config_json['general']['max_epochs'],
            amp_level=config_json['general']['amp_level'],  
            precision=config_json['general']['precision'],  
            accumulate_grad_batches=config_json['general']['grad_acc'],
            deterministic=True,
            check_val_every_n_epoch=1,
        )

        # Train Model
        if args.train_or_test == 'train':
            trainer.fit(model=model, datamodule=datamodule)
        
        # Test Model
        else:
            model_checkpoint = os.path.join(args.checkpoint_dir, f"best_model_kf{fold}.ckpt")
            model = model.load_from_checkpoint(model_checkpoint, map_location=device)
            trainer.test(model=model, datamodule=datamodule)
