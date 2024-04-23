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



### Patch (and Segmentation Mask) Creation using CLAM
To obtain the patches of the good-quality WSIs, using the CLAM framework, you can run the following script:
```bash
#!/bin/bash

echo 'Started CLAM (create_patches_fp.py) on TCGA-BRCA Database.'

# Using CLAM + HistoQC Segmentation and Patch Pipeline
python code/preprocessing/patch_and_segmentation/create_patches_fp.py --source_dir 'data/TCGA-BRCA' --dataset_name 'TCGA-BRCA' --tcgabrca_expstr 'DiagnosticSlide' --save_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC' --patch_size 256 --preset 'code/preprocessing/patch_and_segmentation/presets/tcga.csv' --patch --stitch --use_histoqc_quality_file 'results/HistoQC/TCGA-BRCA/mmxbrcp/hqc_quality_file.csv' --use_histoqc_seg_masks --verbose

python code/preprocessing/patch_and_segmentation/create_patches_fp.py --source_dir 'data/TCGA-BRCA' --dataset_name 'TCGA-BRCA' --tcgabrca_expstr 'TissueSlide' --save_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC' --patch_size 256 --preset 'code/preprocessing/patch_and_segmentation/presets/tcga.csv' --patch --stitch --use_histoqc_quality_file 'results/HistoQC/TCGA-BRCA/mmxbrcp/hqc_quality_file.csv' --use_histoqc_seg_masks --verbose

echo 'Finished CLAM (create_patches_fp.py) on TCGA-BRCA Database.'
```

On the other hand, you can also ignore the HistoQC results and run the following script:
```bash
#!/bin/bash

echo 'Started CLAM (create_patches_fp.py) on TCGA-BRCA Database.'

# Using CLAM: Segmentation and Patch Pipeline
python code/preprocessing/patch_and_segmentation/create_patches_fp.py --source_dir 'data/TCGA-BRCA' --dataset_name 'TCGA-BRCA' --tcgabrca_expstr 'DiagnosticSlide' --save_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationCLAM' --patch_size 256 --preset 'code/preprocessing/patch_and_segmentation/presets/tcga.csv' --seg --save_mask --patch --stitch --verbose

python code/preprocessing/patch_and_segmentation/create_patches_fp.py --source_dir 'data/TCGA-BRCA' --dataset_name 'TCGA-BRCA' --tcgabrca_expstr 'TissueSlide' --save_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationCLAM' --patch_size 256 --preset 'code/preprocessing/patch_and_segmentation/presets/tcga.csv' --seg --save_mask --patch --stitch --verbose

echo 'Finished CLAM (create_patches_fp.py) on TCGA-BRCA Database.'
```



### Feature Extraction 
#### Using CLAM
To perform feature extraction using CLAM, you can run the following script:
```bash
#!/bin/bash

echo 'Started CLAM (extract_features_clam.py) on TCGA-BRCA Database.'

python code/feature_extraction/extract_features_clam.py --gpu_id 0 --data_h5_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC' --process_list_csv_path 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/process_list_autogen.csv' --feat_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam' --batch_size 512 --num_workers 10 --pin_memory --verbose

python code/feature_extraction/extract_features_clam.py --gpu_id 0 --data_h5_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC' --process_list_csv_path 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/process_list_autogen.csv' --feat_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam' --batch_size 512 --num_workers 10 --pin_memory --verbose

echo 'Finished CLAM (extract_features_fp.py) on TCGA-BRCA Database.'

```

#### Using PLIP
To perform feature extraction using PLIP, you can run the following script:
```bash
#!/bin/bash

echo 'Started feature extraction using PLIP on TCGA-BRCA Database.'

python code/feature_extraction/extract_features_plip.py --gpu_id 0 --data_h5_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC' --process_list_csv_path 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/process_list_autogen.csv' --feat_dir 'results/PLIP/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/plip' --batch_size 4096 --num_workers 12 --pin_memory --verbose

python code/feature_extraction/extract_features_plip.py --gpu_id 0 --data_h5_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC' --process_list_csv_path 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/process_list_autogen.csv' --feat_dir 'results/PLIP/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/plip' --batch_size 4096 --num_workers 12 --pin_memory --verbose

echo 'Finished feature extraction using PLIP on TCGA-BRCA Database.'

```