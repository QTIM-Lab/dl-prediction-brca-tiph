# Imports
from __future__ import print_function
import os
import argparse
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch



# Function: See the seed for reproducibility purposes
def set_seed(seed=42):

    # Random Seed
    random.seed(seed)

    # Environment Variable Seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy Seed
    np.random.seed(seed)

    # PyTorch Seed(s)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return



# Method: Load TCGA_BRCA_ssGSEA_Scores
def load_tcga_brca_ssgsea_scores(base_data_path):

    # Read CSV
    df = pd.read_csv(os.path.join(base_data_path, 'Annotations', 'TCGA_BRCA_ssGSEA_Scores.csv'))
    
    # Create a data dictionary
    ssgsea_scores_dict = dict()
    for col_name in list(df.columns):
        if col_name == 'Unnamed: 0':
            ssgsea_scores_dict['label_names'] = df['Unnamed: 0'].values
        else:
            ssgsea_scores_dict[col_name] = df[col_name].values

    return ssgsea_scores_dict



if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(description='CLAM: Model Training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--base_data_path', type=str, required=True, help='Base data path for TCGA-BRCA dataset.')
    args = parser.parse_args()



    # Set seeds (for reproducibility reasons)
    set_seed(seed=args.seed)



    # Get the ssGSEA Scores
    ssgsea_scores_dict = load_tcga_brca_ssgsea_scores(base_data_path=args.base_data_path)
    ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict = dict(), dict()

    for idx, label_ in enumerate(ssgsea_scores_dict['label_names']):
        ssgsea_scores_idx_label_dict[idx] = label_
        ssgsea_scores_label_idx_dict[label_] = idx

    ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict = ssgsea_scores_label_idx_dict, ssgsea_scores_idx_label_dict



    # Pos. Correlation between B-Cell Proliferation, T-Cell Cytotoxicity, Antigen Processing and Presentation
    # Neg. Correlation between Immunosuppression
    # Get task names
    label_names = [
        'gobp_b_cell_proliferation', 
        'gobp_t_cell_mediated_cytotoxicity', 
        'kegg_antigen_processing_and_presentation', 
        'immunosuppression'
    ]

    # Create two NumPy arrays to gather this data
    data_arr = np.zeros((len(ssgsea_scores_dict.keys())-1, len(label_names)))
    data_arr_bin = np.zeros_like(data_arr)
    
    # Go through our dataset
    wsi_idx = 0
    for k, v in ssgsea_scores_dict.items():
        if k not in ('label_names'):
            for l_idx, ln in enumerate(label_names):
                s = v[ssgsea_scores_label_idx_dict[ln]]
                data_arr[wsi_idx, l_idx] = s
                data_arr_bin[wsi_idx, l_idx] = 1 if s > 0 else 0
            wsi_idx += 1
    
    
    
    # Create a DataFrame to compute co-correlation matrix
    # Using continuous data
    data_df = pd.DataFrame(data=data_arr, columns=label_names)
    data_corrmatx = data_df.corr()
    sns.heatmap(data_corrmatx, cmap="Greens", annot=True)
    plt.savefig(
         fname=os.path.join('results/data_analysis', f"task_corr_analysis.png"),
         bbox_inches='tight'
    )
    plt.clf()
    plt.close()


    # Using binary data
    data_df = pd.DataFrame(data=data_arr_bin, columns=label_names)
    data_corrmatx = data_df.corr()
    sns.heatmap(data_corrmatx, cmap="Greens", annot=True)
    plt.savefig(
         fname=os.path.join('results/data_analysis', f"task_corr_bin_analysis.png"),
         bbox_inches='tight'
    )
    plt.clf()
    plt.close()