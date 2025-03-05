# Imports
from __future__ import print_function, division
import pandas as pd
import h5py

# PyTorch Imports
from torch.utils.data import Dataset
from torchvision import transforms



# Class: WSI Bag (of Patches/Tiles) w/ coordinates for CLAM v2 (reads the region of the tile)
class WSISlideBagCLAMFPv2(Dataset):
    
    # Method: __init__
    def __init__(self, file_path, wsi, pretrained=False, custom_downsample=1, target_patch_size=-1, transform=None, verbose=False):

        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """


        # Class variables
        self.pretrained = pretrained
        self.wsi = wsi
        self.verbose = verbose
        self.file_path = file_path
        
        # Pipeline of Transforms
        transforms_ = [
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406) if pretrained else (0.5, 0.5, 0.5), 
                std=(0.229, 0.224, 0.225) if pretrained else (0.5, 0.5, 0.5)
            )
        ]
        if transform is None:
            self.transform = transforms.Compose(transforms_)
        else:
            assert isinstance(transform, list)
            transform.extend(transforms_)
            self.transform = transforms.Compose(transform)


        # Open H5 file with coordinates information to get the length of the dataset
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
        
        
        # Print summary of the dataset
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
                
        print("Dataset attributes items: ", dset.attrs.items())
        print('Using ImageNet pretrained weights and normalization: ', self.pretrained)
        print('Transforms:', self.transform)
        print('Target patch size: ', self.target_patch_size)

        return


    # Method: __getitem__
    def __getitem__(self, idx):

        # Open hdf5 file and get the coordinates information
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]

        # Use these coordinates and convert them into an image
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        # Resize image (if needed)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)

        # Apply transforms
        img = self.transform(img).unsqueeze(0)
        
        return img, coord



# Class: WSI Bag (of Patches/Tiles) w/ coordinates for CLAM v3 (analysis) (reads the region of the tile)
class WSISlideBagCLAMFPv3(Dataset):
    
    # Method: __init__
    def __init__(self, file_path, wsi, custom_downsample=1, target_patch_size=-1, transform=None, verbose=False):

        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """


        # Class variables
        self.wsi = wsi
        self.verbose = verbose
        self.file_path = file_path
        self.transform = transform


        # Open H5 file with coordinates information to get the length of the dataset
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
        
        
        # Print summary of the dataset
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
                
        print("Dataset attributes items: ", dset.attrs.items())
        print('Transforms:', self.transform)
        print('Target patch size: ', self.target_patch_size)

        return


    # Method: __getitem__
    def __getitem__(self, idx):

        # Open hdf5 file and get the coordinates information
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]

        # Use these coordinates and convert them into an image
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        # Resize image (if needed)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)        
        
        return img



# Class: Dataset with All WSI Slide Bags for CLAM
class DatasetAllBagsCLAM(Dataset):

    # Method: __init__
    def __init__(self, csv_path):
        
        # Read dataframe
        self.df = pd.read_csv(csv_path)

        return

    # Method: __len__
    def __len__(self):
        return len(self.df)

    # Method: __getitem__
    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
