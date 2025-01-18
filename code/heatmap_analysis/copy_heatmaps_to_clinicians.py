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



# Go through the checkpoint directories
for cp_dir in CHECKPOINT_DIRS:
    cp_dir_content = os.listdir(cp_dir)

    assert 'config.json' in cp_dir_content
    assert 'heatmaps' in cp_dir_content
    assert 'test_metrics_kf0.csv' in cp_dir_content


    # Load configuration JSON
    with open('config.json', 'r') as j:
        config_json = json.load(j)
    print(config_json)
