# Imports
import argparse
import os
import pandas as pd



if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(description='HistoQC: WSI Quality Control File List.')
    parser.add_argument('--database', type=str, required=True, choices=['TCGABRCAData'], help="The database in which we want to run the HistoQC.")
    parser.add_argument('--histoqc_results_path', type=str, required=True, help="The directory where the HistoQC results are stored.")
    args = parser.parse_args()



    # Create a dictionary for the quality results
    results_dict = {
        'wsi_folder_path':list(),
        'is_good_quality':list()
    }


    # Load database
    if args.database == 'TCGABRCAData':

        # DB Constrains / Assertion
        assert args.histoqc_results_path is not None

        # Get the subfolders of this path
        hqc_r_subdirs = [f for f in os.listdir(args.histoqc_results_path) if os.path.isdir(os.path.join(args.histoqc_results_path, f))]

        # Iterate through each folder
        for dir_ in hqc_r_subdirs:
            
            # Check its contents
            dir_contents = [c for c in os.listdir(os.path.join(args.histoqc_results_path, dir_)) if not c.startswith('.')]
            svs_folder = [c for c in os.listdir(os.path.join(args.histoqc_results_path, dir_)) if os.path.isdir(os.path.join(args.histoqc_results_path, dir_, c))]

            # If it contains a .TSV file for results, then it it is good quality
            if 'results.tsv' in dir_contents:
                results_dict['is_good_quality'].append(True)
            else:
                results_dict['is_good_quality'].append(False)
            
            if len(svs_folder) == 1:
                results_dict['wsi_folder_path'].append(
                    os.path.join(args.histoqc_results_path, dir_, svs_folder[0])
                )
            else:
                results_dict['wsi_folder_path'].append(
                    os.path.join('')
                )
        

        # Save this into a .CSV file
        df = pd.DataFrame(results_dict)
        df.to_csv(os.path.join(args.histoqc_results_path, 'hqc_quality_file.csv'))
