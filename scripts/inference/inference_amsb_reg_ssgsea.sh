#!/bin/bash

echo 'Started CLAM Testing on TCGA-BRCA Database.'




# List of checkpoint directories for AM-SB and AM-MB (CONCH Features)
# These are Dago's Augmented Features Models
CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/micropath/results/augmented/gobp_b_cell_proliferation/2025-02-10_11-33-37' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/gobp_t_cell_mediated_cytotoxicity/2025-02-10_12-57-08' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_angiogenesis/2025-02-10_13-23-05' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_epithelial_mesenchymal_transition/2025-02-10_13-58-25' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_fatty_acid_metabolism/2025-02-10_14-21-51' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_glycolysis/2025-02-10_14-30-02' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_oxidative_phosphorylation/2025-02-10_14-34-38' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/immunosuppression/2025-02-10_14-40-02' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/kegg_antigen_processing_and_presentation/2025-02-10_14-48-01' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/kegg_cell_cycle/2025-02-10_14-52-14')



for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Checkpoint: $checkpoint_dir"
    
    # CONCH Features
    python code/models/clam/inference_model_fp.py \
    --gpu_id 0 \
    --checkpoint_dir $checkpoint_dir \
    --dataset 'TCGA-BRCA' \
    --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA' \
    --experimental_strategy 'All' \
    --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features'
done

echo 'Finished CLAM Testing on TCGA-BRCA Database.'