# Imports
from __future__ import print_function, division
import h5py
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch Imports
from torch.utils.data import Dataset



# Class: WSISlideBagPLIP - WSI Bag (of Patches/Tiles) w/ coordinates for PLIP
class WSISlideBagPLIP(Dataset):


    # Method: __init__
    def __init__(self, file_path, wsi, custom_downsample=1, target_patch_size=-1, plip_preprocessor=None, verbose=False, transform=None):

        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """


        # Class variables
        self.file_path = file_path
        self.wsi = wsi
        self.plip_preprocessor = plip_preprocessor
        self.verbose = verbose
        self.transform = transform


        # Open HDF5 file and get the important attributes
        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, ) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
            else:
                self.target_patch_size = None

        if self.verbose:
            self.summary()
        
        return


    # Method: __len__
    def __len__(self):
        return self.length


    # Method: Get the summary of a WSI
    def summary(self):

        # Open hdf5 file and get the coords information
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        
        # Print the attributes
        for name, value in dset.attrs.items():
            print(name, value)

        # Print the WSI information
        print('\nFeature extraction settings')
        print('Target patch size: ', self.target_patch_size)

        return


    # Method: __getitem__
    def __getitem__(self, idx):

        # Open hdf5 file and get the coordinates information
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]

        # Use these coordinates and convert them into an image
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        # If we have a target patch size
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)

        # Apply data augmentation, if available
        if self.transform:
            img = np.array(img)
            img = self.transform(img)
            img = Image.fromarray(img, mode="RGB")

        # Apply PLIP preprocessing 
        if self.plip_preprocessor:
            img = self.plip_preprocessor(images=img, return_tensors='pt')
            img = img['pixel_values']       

        return img, coord



# Class: Dataset_All_Bags
class Dataset_All_Bags(Dataset):


    # Method: __init__
    def __init__(self, csv_path):
        
        # Read dataframe
        self.df = pd.read_csv(csv_path)


    # Method: __len__
    def __len__(self):
        return len(self.df)


    # Method: __getitem__
    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
