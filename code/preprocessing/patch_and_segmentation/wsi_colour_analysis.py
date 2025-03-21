# Imports
import os
import time
import argparse
import numpy as np
import openslide
import matplotlib.pyplot as plt

# Project Imports
from data_utilities import DatasetAllBagsCLAM, WSISlideBagCLAMFPv3



# Function: Compute WSI Features using CLAM-based approach (with ResNet50)
def wsi_colour(file_path, wsi, verbose=False, custom_downsample=1, target_patch_size=-1):
    
    """
    Args:
        file_path: directory of bag (.h5 file)
        wsi: the path to the WSI (.h5 file)
        model: PyTorch model
        batch_size: batch_size for computing features in batches
        verbose: provide feedback in the command line
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """



    # Read WSI information	
    dataset = WSISlideBagCLAMFPv3(
        file_path=file_path,
        wsi=wsi,
        custom_downsample=custom_downsample,
        target_patch_size=target_patch_size,
        transform=None,
        verbose=verbose,
    )





    # Set hdf5 file mode to write when the processing begins
    histogram_r = np.zeros(256)
    histogram_g = np.zeros_like(histogram_r)
    histogram_b = np.zeros_like(histogram_r)

    for idx in range(len(dataset)):
        
        # Get image
        img = dataset.__getitem__(idx)
        img = np.array(img)
        # print(img.shape)

        # Compute histogram
        histogram_r_i, _ = np.histogram(img[:, :, 0], bins=256, range=(0, 256))
        histogram_g_i, _ = np.histogram(img[:, :, 1], bins=256, range=(0, 256))
        histogram_b_i, _ = np.histogram(img[:, :, 2], bins=256, range=(0, 256))

        histogram_r += histogram_r_i
        histogram_g += histogram_g_i
        histogram_b += histogram_b_i

        # print(histogram_r)
        # print(histogram_g)
        # print(histogram_b)

    return histogram_r, histogram_g, histogram_b



if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(description='WSI Colour Analysis.')
    parser.add_argument('--features_h5_dir', nargs='+', type=str, required=True, help="The directory of the features in .pt format for the TCGA-BRCA dataset.")
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()



    # Go through all the subsets
    d_r, d_g, d_b = np.zeros(256), np.zeros(256), np.zeros(256)
    for p in args.features_h5_dir:

        # Get the file of the .CSV file that contains the list of .h5 files (process_list_csv_path from the previous step)
        csv_path = os.path.join(p, 'process_list_autogen.csv')

        # Generate a list of WSI .h5 files
        bags_dataset = DatasetAllBagsCLAM(csv_path)

        # Iterate through the dataset
        bag_candidate_indices = [i for i in range(len(bags_dataset))]
        # print(len(bag_candidate_indices))
        # print(bag_candidate_indices)
        
        
        for bag_candidate_idx in bag_candidate_indices:

            # Get Slide ID
            slide_file_path = bags_dataset[bag_candidate_idx]
            slide_name = slide_file_path.split('/')[-1]
            slide_ext = slide_name.split('.')[-1]
            slide_ext = f'.{slide_ext}'
            slide_id = slide_name.split(slide_ext)[0]
            
            # Get the .h5 file
            bag_name = slide_id+'.h5'
            h5_file_path = os.path.join(p, 'patches', bag_name)
            # print(h5_file_path)
            assert os.path.exists(h5_file_path)
            

            # Verbose
            if args.verbose:
                print(f'Progress: {bag_candidate_idx+1}/{len(bag_candidate_indices)}')
                print('Slide ID: ', slide_id)



            # Verbose (time elapsed)
            if args.verbose:
                time_start = time.time()
            

            # Open WSI
            # TODO: Updated this in the code afterwards
            # print(slide_file_path)
            # slide_file_path = slide_file_path.replace("/autofs/cluster/qtim/datasets/private/MGH_breast/", "/autofs/space/crater_001/datasets/private/breast_path/MGH_breast/")
            assert os.path.exists(slide_file_path)
            wsi = openslide.open_slide(slide_file_path)
            
            # Get histograms
            r, g, b = wsi_colour(
                file_path=h5_file_path,
                wsi=wsi,
                verbose=args.verbose
            )


            # Build dataset histograms
            d_r += r
            d_g += g
            d_b += b
            

            # Verbose (time elapsed)
            if args.verbose:
                time_elapsed = time.time() - time_start
                print(f'Computing colour histogram for {slide_id} took {time_elapsed}s.')
        

    # Create a figure
    plt.title("Color Histogram")
    plt.xlabel("Color Value")
    plt.xlim([0, 256])
    plt.ylabel("Pixel Count")
    plt.plot([i for i in range(256)], d_r, color='red', label='red')
    plt.plot([i for i in range(256)], d_g, color='green', label='green')
    plt.plot([i for i in range(256)], d_b, color='blue', label='blue')
    plt.legend()
    plt.savefig(
        fname='colour_hist.png',
        bbox_inches='tight'
    )
    plt.clf()
    plt.close()
