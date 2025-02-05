# Imports
import os
import argparse
import pandas as pd



# Method: Obtain Case ID
def get_case_id(wsi_path_or_name, mode='simple'):

    assert mode in ('simple', 'extended')

    # Pipeline to get Case ID
    parsed_path = wsi_path_or_name.split('/')[-1]
    parsed_path = parsed_path.split('.')[0]
    if mode == 'simple':
        parsed_path = parsed_path.split('-')[0:3]
    else:
        parsed_path = parsed_path.split('-')[0:4]

    # Get CID
    case_id = '-'.join(parsed_path)

    return case_id



if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(description='CLAM: Heatmap Generation (for Clinical Study).')
    parser.add_argument('--clinicians_dir', type=str, required=True, help="The directory where we will save these studies for the clinicians.")
    parser.add_argument('--researchers_dir', type=str, required=True, help="The directory where we will save these studies for the researchers.")
    parser.add_argument('--task', type=str, required=True, choices=["b_cell_proliferation", "t_cell_mediated_cytotoxicity"], help="The annotation file will be generated to this task.")
    parser.add_argument('--verbose', action="store_true", help="Print execution information.")
    args = parser.parse_args()



    # Create a dictionary with data
    data_dict = {
        "task":list(),
        "patient":list(),
        "label":list(),
        "wsi":list(),
        "set":list(),
        "img":list(),
        "img_path":list(),
        "annotation":list()
    }

    # Open the task directory
    task_dir = os.path.join(args.clinicians_dir, args.task)
    # print(task_dir)
    
    # Go through labels
    for label in ["neg", "pos"]:
        label_dir = os.path.join(task_dir, label)
        # print(label_dir)

        # Get WSIs
        wsi_dirs = os.listdir(label_dir)
        wsi_dirs = [d for d in wsi_dirs if os.path.isdir(os.path.join(label_dir, d))]

        # Go through WSI directories
        for wsi in wsi_dirs:
            wsi_dir_ = os.path.join(label_dir, wsi)
            patient = get_case_id(wsi)

            # Get the set folders
            set_dirs = os.listdir(wsi_dir_)
            set_dirs = [d for d in set_dirs if os.path.isdir(os.path.join(wsi_dir_, d))]

            # Go through set directories
            for set_ in set_dirs:
                set_dir_ = os.path.join(wsi_dir_, set_)

                # Get the images of this set
                set_imgs = os.listdir(set_dir_)
                set_imgs = [i for i in set_imgs if i.endswith('.png')]

                # Go through these set images
                for img in set_imgs:
                    img_path = os.path.join(set_dir_, img)

                    # Append data 
                    data_dict["task"].append(args.task)
                    data_dict["patient"].append(patient)
                    data_dict["label"].append(label)
                    data_dict["wsi"].append(wsi)
                    data_dict["set"].append(set_)
                    data_dict["img"].append(img)
                    data_dict["img_path"].append(img_path)
                    data_dict["annotation"].append("")
                    # print(data_dict)



    # Convert Dictionary into a DataFrame
    data_df = pd.DataFrame.from_dict(data_dict)
    print(data_df)

    # Save a CSV
    data_df.to_csv(os.path.join(task_dir, "global_annotation.csv"))
