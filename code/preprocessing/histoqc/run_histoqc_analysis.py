# Imports
import argparse
import os
import subprocess
import pandas as pd
from tqdm import tqdm



# Class: TCGABRCAData
class TCGABRCAData:
    def __init__(self, base_data_path='TCGA-BRCA', experimental_strategy='All'):
        
        assert experimental_strategy in ('All', 'DiagnosticSlide', 'TissueSlide')

        # Class variables
        self.base_data_path = base_data_path
        self.experimental_strategy = experimental_strategy

        # Choose Experimental Strategy
        if experimental_strategy in ('DiagnosticSlide', 'TissueSlide'):
            if experimental_strategy == 'DiagnosticSlide':
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'DiagnosticSlide.csv'))
                assert len(data) == 1133, f'The size of the Diagnostic Slide subset should be 1133 and not {len(data)}.'
            else:
                data = pd.read_csv(os.path.join(base_data_path, 'ExperimentalStrategy', 'TissueSlide.csv'))
                assert len(data) == 1978, f'The size of the Tissue Slide subset should be 1978 and not {len(data)}.'
            
            # Get folders and SVS files
            folders, svs_files = data['folder_name'].values, data['svs_name']

            # Create a list for the SVS filepaths
            svs_fpaths = list()

            # Go through each folder subset and append the filepath to the list
            for folder, svs_file in zip(folders, svs_files):
                svs_fpath = os.path.join(base_data_path, 'WSI', folder, svs_file)
                svs_fpaths.append(svs_fpath)

        # Or get all data samples
        else:
            svs_fpaths = self.get_all_folders_and_svs_fpaths()

        self.svs_fpaths = svs_fpaths

        return


    # Method: get_all_folders_and_svs
    def get_all_folders_and_svs_fpaths(self):

        # Enter the WSI directory of the dataset
        wsi_directory = os.path.join(self.base_data_path, 'WSI')

        # Get folders
        folders = [f for f in os.listdir(wsi_directory) if not f.startswith('.')]

        # Create a list for the SVS filepaths
        svs_fpaths = list()

        # Enter each folder
        for folder in folders:

            # Get the contents of each folder
            folder_contents = [c for c in os.listdir(os.path.join(wsi_directory, folder)) if not c.startswith('.')]
            
            # Get the SVS file(s)
            svs_files = [s for s in folder_contents if s.endswith('svs')]

            # Build SVS filepaths
            for svs_f in svs_files:
                svs_fpath = os.path.join(wsi_directory, folder, svs_f)

                # Append it to the list
                svs_fpaths.append(svs_fpath)

        return svs_fpaths


    # Method: Create a dictionary of svs filepaths connected to CIDs
    def get_svs_fpaths_dict(self):

        # Create a dictionary
        svs_fpaths_dict = dict()

        # Go through the svs_fpaths
        for path in self.svs_fpaths:

            # Parse names
            parsed_path = path.split('/')[-1]
            parsed_path = parsed_path.split('.')[0]
            parsed_path = parsed_path.split('-')[0:4]

            # Get CID
            cid = '-'.join(parsed_path)

            # Add to dictionary
            svs_fpaths_dict[cid] = path

        return svs_fpaths_dict



if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(description='HistoQC: WSI Quality Control.')
    parser.add_argument('--database', type=str, required=True, choices=['TCGABRCAData'], help="The database in which we want to run the HistoQC.")
    parser.add_argument('--base_data_path', type=str, help="The base data path. Applies to the following databases: TCGABRCAData.")
    parser.add_argument('--experimental_strategy', type=str, choices=['All', 'DiagnosticSlide', 'TissueSlide'], help="The experimental strategy. Applies to: TCGABRCAData.")
    parser.add_argument('--outdir', type=str, required=True, help="The output directory for the HistoQC results.")
    args = parser.parse_args()



    # Load database
    if args.database == 'TCGABRCAData':

        # DB Constrains / Assertion
        assert args.base_data_path is not None
        assert args.experimental_strategy is not None

        database = TCGABRCAData(
            base_data_path=args.base_data_path,
            experimental_strategy=args.experimental_strategy
        )

        wsi_full_paths = database.svs_fpaths
        wsi_dirs = list()
        for wsi_path in wsi_full_paths:
            wsi_path_ = wsi_path.split('/')[0:-1]
            wsi_path_ = '/'.join(wsi_path_)
            wsi_dirs.append(wsi_path_)


    # Run HistoQC on each directory
    for wsi_dir in tqdm(wsi_dirs):
        wsi_folder = wsi_dir.split('/')[-1]
        result = subprocess.run(
            ["python", '-m', "histoqc", '--config', 'v2.1', '--nprocesses', '10', f'{wsi_dir}/*.svs', '--outdir', f'{args.outdir}/{wsi_folder}', '--force'],
            capture_output=True,
            text=True,
            check=True
        )
