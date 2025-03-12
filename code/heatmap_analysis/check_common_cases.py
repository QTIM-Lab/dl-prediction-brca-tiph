# Imports
import os
import pandas as pd



# List of paths to assess
# angiogenesis, epithelial-mesenchymal transition, cell cycling, immunosuppression, t-cell mediated cytotoxicity
paths_to_assess = [
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/2024-04-25_21-05-55/heatmaps',
    '/autofs/space/crater_001/projects//breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-04-25_09-13-05/heatmaps',
    '/autofs/space/crater_001/projects//breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-04-25_11-05-38/heatmaps',
    '/autofs/space/crater_001/projects//breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/2024-04-26_01-28-02/heatmaps',
    '/autofs/space/crater_001/projects//breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/2024-04-29_03-31-36/heatmaps'
]



# Get the split
case_counts = dict()

# Iterate through the data in the dataframes
for path_ in paths_to_assess:

    # Get values
    cases = [c for c in os.listdir(path_) if not c.startswith('.')]
    # print(cases)

    # Count these values
    for case_ in cases:
        if case_ not in case_counts.keys():
            case_counts[case_] = 1
        else:
            case_counts[case_] += 1


# Get image names that are common to the n CSVs that we loaded
image_counts_inv = dict()
for img_name, img_count in case_counts.items():
    # print(img_name, img_count)
    if img_count not in image_counts_inv.keys():
        image_counts_inv[img_count] = list()
        image_counts_inv[img_count].append(img_name)
    else:
        image_counts_inv[img_count].append(img_name)

img_count_values = [k for k in image_counts_inv.keys()]
print(img_count_values)
for cnt in img_count_values:
    cnt_dict = {cnt:list()}
    for i_cnt, i_name in image_counts_inv.items():
        if i_cnt == cnt:
            cnt_dict[cnt].append(i_name)
    cnt_dict_df = pd.DataFrame.from_dict(cnt_dict)
    cnt_dict_df.to_csv(f"common_results_idx{cnt}.csv", index=False)
