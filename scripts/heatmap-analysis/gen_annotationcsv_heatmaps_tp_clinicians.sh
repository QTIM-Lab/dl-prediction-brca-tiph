#!/bin/bash

echo 'Generating Global Annotation CSV file...'



# All Tasks
# TASKS=('b_cell_proliferation' \
#  't_cell_mediated_cytotoxicity' \
#  'angiogenesis' \
#  'epithelial_mesenchymal_transition')

# B-Cell Proliferation
# TASKS=('b_cell_proliferation')

# T-Cell Mediated Toxicity
# TASKS=('t_cell_mediated_cytotoxicity')

# Angiogenesis
# TASKS=('angiogenesis')

# Epithelial Mesenchymal Transition
TASKS=('epithelial_mesenchymal_transition')



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