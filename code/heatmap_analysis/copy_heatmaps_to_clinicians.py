# Imports
import os
import pandas as pd
import json



# Get the checkpoints directories
CHECKPOINT_DIRS = [
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/2024-04-25_23-07-04',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/2024-04-25_21-05-55',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-04-25_09-13-05',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-04-25_11-05-38',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/2024-04-25_13-01-04',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/2024-04-25_17-12-36',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/2024-04-25_15-14-31',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/2024-04-29_03-31-36',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/2024-04-25_19-10-32',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/2024-04-26_01-28-02'
]



# Define the clinicians directory
CLINICIANS_DIR = '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/'



# Dictionary to map names of the tasks
names_dict = {
    'gobp_b_cell_proliferation':'b_cell_proliferation',
    'gobp_t_cell_mediated_cytotoxicity':'t_cell_mediated_cytotoxicity',
    'hallmark_angiogenesis':'angiogenesis',
    'hallmark_epithelial_mesenchymal_transition':'epithelial_mesenchymal_transition',
    'hallmark_fatty_acid_metabolism':'fatty_acid_metabolism',
    'hallmark_glycolysis':'glycolysis',
    'hallmark_oxidative_phosphorylation':'oxidative_phosphorylation',
    'immunosuppression':'immunosuppression',
    'kegg_antigen_processing_and_presentation':'antigen_processing_and_presentation',
    'kegg_cell_cycle':'cell_cycle'
}

# Dictionary to map labels
labels_dict = {
    0:'neg',
    1:'pos'
}



# Go through the checkpoint directories
for cp_dir in CHECKPOINT_DIRS:
    cp_dir_content = os.listdir(cp_dir)
    # print(cp_dir_content)
    task_ = cp_dir.split('/')[-2]
    # print(task_)
    task = names_dict[task_]
    # print(task)

    # Configuration JSON
    # assert 'config.json' in cp_dir_content
    # with open(os.path.join(cp_dir, 'config.json'), 'r') as j:
    #     config_json = json.load(j)
    # print(config_json)

    # Test Metrics Information
    # assert 'test_metrics_kf0.csv' in cp_dir_content
    # test_metrics_df = pd.read_csv(os.path.join(cp_dir, 'test_metrics_kf0.csv'))
    # print(test_metrics_df)
    
    # Test Set Information
    # assert 'test_set_kf0.csv' in cp_dir_content
    # test_set_df = pd.read_csv(os.path.join(cp_dir, 'test_set_kf0.csv'))
    # print(test_set_df)

    # Heatmaps
    assert 'heatmaps' in cp_dir_content
    heatmaps_dir = os.path.join(cp_dir, 'heatmaps')
    heatmaps_dir_content = os.listdir(heatmaps_dir)
    # print(heatmaps_dir_content)
    for wsi_hmaps_dir in heatmaps_dir_content:
        wsi_hmaps_dir_content = os.listdir(os.path.join(heatmaps_dir, wsi_hmaps_dir))
        if len(wsi_hmaps_dir_content) > 0:
            assert 'info.csv' in wsi_hmaps_dir_content
            wsi_hmap_info_df = pd.read_csv(os.path.join(heatmaps_dir, wsi_hmaps_dir, 'info.csv'))
            # print(wsi_hmap_info_df)
            label_ = wsi_hmap_info_df['label'].values[0]
            label = labels_dict[label_]
            print(label_, label)

    exit()