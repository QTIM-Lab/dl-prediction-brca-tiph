#!/bin/bash



echo "Started SSGSEA Task Correlation Analysis"
python code/data_analysis/task_corr_analysis.py --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA'
echo "Finished SSGSEA Task Correlation Analysis"