#!/bin/bash



python code/preprocessing/patch_and_segmentation/wsi_colour_analysis.py \
 --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC' \
 --verbose