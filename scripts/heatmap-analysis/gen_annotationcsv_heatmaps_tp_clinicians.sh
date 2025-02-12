#!/bin/bash

echo 'Generating Global Annotation CSV file...'





# B-Cell Proliferation
# TASKS=('b_cell_proliferation')

# T-Cell Mediated Toxicity
TASKS=('t_cell_mediated_cytotoxicity')



# T-Cell Cytotoxicity, Angiogenesis, Epithelial-Mesenchymal Transition
# CHECKPOINT_DIRS=(
 # '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-04-25_09-13-05' \
 # '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-04-25_11-05-38')



for task in "${TASKS[@]}"
do
    echo "Task: $task"
    
    # CLAM Features
    python code/heatmap_analysis/gen_annotationcsv_heatmaps_tp_clinicians.py \
    --clinicians_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/clinicians' \
    --researchers_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/researchers' \
    --task $task
done



echo 'Finished.'