# Deep Learning-based Prediction of Breast Cancer Tumor and Immune Phenotypes from Histopathology
Implementation of the paper "Deep Learning-based Prediction of Breast Cancer Tumor and Immune Phenotypes from Histopathology" by Tiago Gonçalves, Dagoberto Pulido-Arias, Julian Willett, Katharina V. Hoebel, Mason Cleveland, Syed Rakin Ahmed, Elizabeth Gerstner, Jayashree Kalpathy-Cramer, Jaime S. Cardoso, Christopher P. Bridge and Albert E. Kim.



# Data Preparation
## Preprocessing
### HistoQC Analysis
You can start by running the HistoQC package to obtain a list of the good-quality WSIs. 
To run the HistoQC package on the data, you can run the following script:
```bash
#!/bin/bash
echo 'Started HistoQC on TCGA-BRCA Database.'
cd code/preprocessing/histoqc
python run_histoqc_analysis.py --database 'TCGABRCAData' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --outdir 'results/HistoQC/TCGA-BRCA/mmxbrcp'
echo 'Finished HistoQC on TCGA-BRCA Database.'
```

To generate the `.CSV` with the list of good-quality WSIs, you can run the following script:
```bash
#!/bin/bash
echo 'Started HistoQC Quality File on TCGA-BRCA Database.'
python code/preprocessing/histoqc/generate_quality_file_list.py --database 'TCGABRCAData' --histoqc_results_path 'results/HistoQC/TCGA-BRCA/mmxbrcp'
echo 'Finished HistoQC Quality File on TCGA-BRCA Database.'
```