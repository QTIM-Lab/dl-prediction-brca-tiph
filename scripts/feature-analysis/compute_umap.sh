#!/bin/bash



# Environment Variables
export PYENV_VERSION=mmxbrcp
export LD_LIBRARY_PATH="/autofs/space/crater_001/tools/usr/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=GPU-a6f9db4e-0b70-8386-34d7-46f9a00b32e9



echo 'Started UMAP on Regression Model Training on TCGA-BRCA Database.'



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
    python code/feature_analysis/conch/compute_umap.py \
    --gpu_id 0 \
    --seed 42 \
    --checkpoint_dir $checkpoint_dir \
    --dataset 'TCGA-BRCA' \
    --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA' \
    --experimental_strategy 'All' \
    --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features'
done



echo 'Finished UMAP on Regression Model Training on TCGA-BRCA Database.'