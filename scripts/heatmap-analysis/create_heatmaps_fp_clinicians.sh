#!/bin/bash
#SBATCH --account=qtim
#SBATCH --partition=rtx6000,rtx8000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=chpa         # Job name
#SBATCH -o chpa.out             # STDOUT
#SBATCH -e chpa.err             # STDERR
#SBATCH -M all
#SBATCH --mail-type=ALL


export PYENV_VERSION=mmxbrcp
export LD_LIBRARY_PATH="/autofs/space/crater_001/tools/usr/lib64:${LD_LIBRARY_PATH}"


echo 'Started CLAM (create_heatmaps_fp.py) on TCGA-BRCA Database.'

# List of checkpoint directories for AM-SB and AM-MB (CLAM/ResNet50 Features)
# CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/2024-04-25_23-07-04' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/2024-04-25_21-05-55' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-04-25_09-13-05' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-04-25_11-05-38' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/2024-04-25_13-01-04' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/2024-04-25_17-12-36' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/2024-04-25_15-14-31' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/2024-04-29_03-31-36' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/2024-04-25_19-10-32' \
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/2024-04-26_01-28-02')



# B-Cell Proliferation
# CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/2024-04-25_23-07-04')



# T-Cell Cytotoxicity, Angiogenesis, Epithelial-Mesenchymal Transition
# CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/2024-04-25_21-05-55' \
 # '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-04-25_09-13-05' \
 # '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-04-25_11-05-38')

# Angiogenesis, Epithelial-Mesenchymal Transition
CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-04-25_09-13-05' \
 '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-04-25_11-05-38')



for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Started CLAM Heatmap Generation for checkpoint: $checkpoint_dir"
    
    # CLAM Features
    python code/models/clam/create_heatmaps_fp_clinicians.py \
    --clinicians_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/clinicians' \
    --researchers_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/researchers' \
    --gpu_id -1 \
    --checkpoint_dir $checkpoint_dir \
    --dataset 'TCGA-BRCA' \
    --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA' \
    --experimental_strategy 'All' \
    --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features' \
    --patches_from 'CLAM' \
    --generate_heatmaps_for 'test' \
    --heatmap_config_file 'code/models/clam/config/tcgabrca_clam_fts_am_sb_heatmap_config.json' \
    --use_histoqc_quality_file '/autofs/space/crater_001/projects/breast-cancer-pathology/results/HistoQC/TCGA-BRCA/mmxbrcp/hqc_quality_file.csv' \
    --use_histoqc_seg_masks \
    --verbose

    echo "Finished CLAM Heatmap Generation for checkpoint: $checkpoint_dir"
done

echo 'Finished CLAM Heatmap Generation on TCGA-BRCA Database.'