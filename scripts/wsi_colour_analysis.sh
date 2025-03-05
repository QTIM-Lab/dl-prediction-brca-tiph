#!/bin/bash



python code/preprocessing/patch_and_segmentation/wsi_colour_analysis.py \
 --use_histoqc_quality_file '/autofs/space/crater_001/projects/breast-cancer-pathology/results/HistoQC/TCGA-BRCA/mmxbrcp/hqc_quality_file.csv' \
 --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features' \
 --verbose