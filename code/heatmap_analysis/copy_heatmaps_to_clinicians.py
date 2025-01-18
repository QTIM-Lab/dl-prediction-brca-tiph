# Imports
import os
import pandas as pd



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



# Go through the checkpoint directories
for cp_dir in CHECKPOINT_DIRS:
    cp_dir_content = os.listdir(cp_dir)
    print(cp_dir_content)
