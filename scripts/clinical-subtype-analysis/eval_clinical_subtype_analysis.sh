#!/bin/bash



# Environment Variables
export PYENV_VERSION=mmxbrcp
export LD_LIBRARY_PATH="/autofs/space/crater_001/tools/usr/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=MIG-752466c4-c584-5e2c-9402-e840a3cf5e6f



echo 'Started Inference Info about Clinical Subtype TCGA-BRCA Database.'

# List of checkpoint directories for AM_SB (CLAM/ResNet50 Features)
CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-11-02_13-45-43'\
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-11-02_15-07-48'\
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/2024-11-02_16-26-16'\
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/2024-11-02_17-46-30'\
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/2024-11-02_18-59-12'\
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/2024-11-02_20-13-27'\
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/2024-11-02_21-28-36'\
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/2024-11-02_22-42-27'\ 
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/2024-11-03_00-01-46'\
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/2024-11-03_01-16-59')

 for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Checkpoint: $checkpoint_dir"

    python code/clinical-subtype-analysis/eval_clinical_subtype_analysis.py \
    --gpu_id 0 \
    --checkpoint_dir $checkpoint_dir
done

echo 'Finished Inference Info about Clinical Subtype TCGA-BRCA Database.'