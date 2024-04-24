# Imports
import copy

# PyTorch & PyTorch Ligthning Imports
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Project Imports
from data_utilities_datasets import TCGABRCA_MIL_Dataset



# Class: DataInterface
class DataInterface(pl.LightningDataModule):

    # Method: __init__
    def __init__(self, batch_size=1, num_workers=4, pin_memory=True, dataset_name=None, fold=None, fusion_strategy=None, setting=None, dataset_args=None):       
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_name = dataset_name
        self.fold = fold
        self.fusion_strategy = fusion_strategy
        self.setting = setting
        self.dataset_args = dataset_args

        return

 
    # Method: prepare_data
    def prepare_data(self):
        pass


    # Method: setup
    def setup(self, stage=None):

        # Load DataSet
        if self.dataset_name == "TCGA-BRCA":
                dataset = TCGABRCA_MIL_Dataset(**self.dataset_args)


        # Get Train, Validation or Test sets
        if stage == 'fit' or stage is None:
            
            # Train
            self.train_dataset = copy.deepcopy(dataset)
            self.train_dataset.select_split(split='train')
            self.train_dataset.select_fold(fold=self.fold)
            
            # Validation
            self.val_dataset = copy.deepcopy(dataset)
            self.val_dataset.select_split(split='validation')
            self.val_dataset.select_fold(fold=self.fold)
 
        if stage == 'test' or stage is None:

            # Test
            self.test_dataset = copy.deepcopy(dataset)
            self.test_dataset.select_split(split='test')
            self.test_dataset.select_fold(fold=self.fold)
        
        return


    # Method: train_dataloader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)


    # Method: val_dataloader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)


    # Method: test_dataloader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False)
