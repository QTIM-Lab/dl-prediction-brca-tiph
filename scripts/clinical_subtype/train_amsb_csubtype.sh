#!/bin/bash
#SBATCH --account=qtim
#SBATCH --partition=rtx6000,rtx8000
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=train_amsb_csubtype         # Job name
#SBATCH -o train_amsb_csubtype.out             # STDOUT
#SBATCH -e train_amsb_csubtype.err             # STDERR
#SBATCH -M all
#SBATCH --mail-type=ALL



export PYENV_VERSION=mmxbrcp
export LD_LIBRARY_PATH="/autofs/space/crater_001/tools/usr/lib64:${LD_LIBRARY_PATH}"



echo 'Started AM-SB (Clinical Subtype) Training on TCGA-BRCA Database.'




# Train
python code/models/clam/train_val_model_fp.py \
 --gpu_id 0 \
 --results_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints' \
 --dataset 'TCGA-BRCA' \
 --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA' \
 --experimental_strategy 'All' \
 --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features' \
 --config_json 'code/models/clam/config/clinical_subtype/tcgabrca_conch_fts_am_sb_config.json'



echo 'Finished AM-SB (Clinical Subtype) Training on TCGA-BRCA Database.'