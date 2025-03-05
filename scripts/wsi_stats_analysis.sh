#!/bin/bash



python code/models/clam/inference_model_fp.py \
 --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA' \
 --use_histoqc_quality_file '/autofs/space/crater_001/projects/breast-cancer-pathology/results/HistoQC/TCGA-BRCA/mmxbrcp/hqc_quality_file.csv' \
 --verbose