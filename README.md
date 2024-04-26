# Deep Learning-based Prediction of Breast Cancer Tumor and Immune Phenotypes from Histopathology
Implementation of the paper "Deep Learning-based Prediction of Breast Cancer Tumor and Immune Phenotypes from Histopathology" by Tiago Gonçalves, Dagoberto Pulido-Arias, Julian Willett, Katharina V. Hoebel, Mason Cleveland, Syed Rakin Ahmed, Elizabeth Gerstner, Jayashree Kalpathy-Cramer, Jaime S. Cardoso, Christopher P. Bridge and Albert E. Kim.

[paper](https://arxiv.org/abs/2404.16397) | [poster](poster.pdf)

# Abstract
The interactions between tumor cells and the tumor microenvironment (TME) dictate therapeutic efficacy of radiation and many systemic therapies in breast cancer. However, to date, there is not a widely available method to reproducibly measure tumor and immune phenotypes for each patient's tumor. Given this unmet clinical need, we applied multiple instance learning (MIL) algorithms to assess activity of ten biologically relevant pathways from the hematoxylin and eosin (H&E) slide of primary breast tumors. We employed different feature extraction approaches and state-of-the-art model architectures. Using binary classification, our models attained area under the receiver operating characteristic (AUROC) scores above 0.70 for nearly all gene expression pathways and on some cases, exceeded 0.80. Attention maps suggest that our trained models recognize biologically relevant spatial patterns of cell sub-populations from H&E. These efforts represent a first step towards developing computational H&E biomarkers that reflect facets of the TME and hold promise for augmenting precision oncology.

![](abstract_fig.png)


# Data Preparation
## Preprocessing
### HistoQC Analysis
We start by running the HistoQC package to obtain a list of the good-quality WSIs. 

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
Then, we obtain the patches (and segmentation masks) from the WSIs.

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



## Feature Extraction
Next, we can proceed into feature extraction.

### Using CLAM
To perform feature extraction using CLAM, you can run the following script:
```bash
#!/bin/bash

echo 'Started CLAM (extract_features_clam.py) on TCGA-BRCA Database.'

python code/feature_extraction/extract_features_clam.py --gpu_id 0 --data_h5_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC' --process_list_csv_path 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/process_list_autogen.csv' --feat_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam' --batch_size 512 --num_workers 10 --pin_memory --verbose

python code/feature_extraction/extract_features_clam.py --gpu_id 0 --data_h5_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC' --process_list_csv_path 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/process_list_autogen.csv' --feat_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam' --batch_size 512 --num_workers 10 --pin_memory --verbose

echo 'Finished CLAM (extract_features_fp.py) on TCGA-BRCA Database.'

```

### Using PLIP
To perform feature extraction using PLIP, you can run the following script:
```bash
#!/bin/bash

echo 'Started feature extraction using PLIP on TCGA-BRCA Database.'

python code/feature_extraction/extract_features_plip.py --gpu_id 0 --data_h5_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC' --process_list_csv_path 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/process_list_autogen.csv' --feat_dir 'results/PLIP/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/plip' --batch_size 4096 --num_workers 12 --pin_memory --verbose

python code/feature_extraction/extract_features_plip.py --gpu_id 0 --data_h5_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC' --process_list_csv_path 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/process_list_autogen.csv' --feat_dir 'results/PLIP/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/plip' --batch_size 4096 --num_workers 12 --pin_memory --verbose

echo 'Finished feature extraction using PLIP on TCGA-BRCA Database.'
```



# Models 
## Training
Finally, we can move forward to model training.

### Using CLAM framework
To train the AM-SB/AM-MB models, you can run the following script:
```bash
#!/bin/bash

echo 'Started CLAM (train_model_fp.py) on TCGA-BRCA Database.'

# List of labels for this project
LABELS=('hallmark_angiogenesis'\
 'hallmark_epithelial_mesenchymal_transition'\
 'hallmark_fatty_acid_metabolism'\
 'hallmark_oxidative_phosphorylation'\
 'hallmark_glycolysis'\
 'kegg_antigen_processing_and_presentation'\
 'gobp_t_cell_mediated_cytotoxicity'\
 'gobp_b_cell_proliferation'\
 'kegg_cell_cycle'\
 'immunosuppression')


for label in "${LABELS[@]}"
do
    echo "Started CLAM (train_model_fp.py) for label: $label"
    
    # CLAM (ResNet50) Features
    python code/models/clam/train_val_model_fp.py --gpu_id 0 --results_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints' --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam/pt_files' 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam/pt_files' --label $label --config_json 'code/models/clam/config/tcgabrca_clam_fts_am_sb_config.json'
    
    python code/models/clam/train_val_model_fp.py --gpu_id 0 --results_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints' --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam/pt_files' 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam/pt_files' --label $label --config_json 'code/models/clam/config/tcgabrca_clam_fts_am_mb_config.json'
    
    # PLIP Features
    python code/models/clam/train_val_model_fp.py --gpu_id 0 --results_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints' --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/PLIP/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/plip/pt_files' 'results/PLIP/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/plip/pt_files' --label $label --config_json 'code/models/clam/config/tcgabrca_plip_fts_am_sb_config.json'
    
    python code/models/clam/train_val_model_fp.py --gpu_id 0 --results_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints' --dataset 'TCGA-BRCA' --base_data_path 'results/PLIP/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/plip/pt_files' 'results/PLIP/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/plip/pt_files' --label $label --config_json 'code/models/clam/config/tcgabrca_plip_fts_am_mb_config.json'

    echo "Finished CLAM (train_model_fp.py) for label: $label"
done

echo 'Finished CLAM (train_model_fp.py) on TCGA-BRCA Database.'

```

### Using TransMIL
To train the TransMIL models, you can run the following script:
```bash
#!/bin/bash

echo 'Started TransMIL (train_model_fp.py) on TCGA-BRCA Database.'

# List of labels for this project
LABELS=('hallmark_angiogenesis'\
 'hallmark_epithelial_mesenchymal_transition'\
 'hallmark_fatty_acid_metabolism'\
 'hallmark_oxidative_phosphorylation'\
 'hallmark_glycolysis'\
 'kegg_antigen_processing_and_presentation'\
 'gobp_t_cell_mediated_cytotoxicity'\
 'gobp_b_cell_proliferation'\
 'kegg_cell_cycle'\
 'immunosuppression')


for label in "${LABELS[@]}"
do
    echo "Started TransMIL (train_model_fp.py) for label: $label"
    
    # CLAM (ResNet50) Features
    python code/models/transmil/train_test_model_fp.py --gpu_id 0 --results_dir 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints' --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam/pt_files' 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam/pt_files' --label $label --config_json 'code/models/transmil/config/tcgabrca_clam_fts_transmil_config.json' --train_or_test 'train'
    
    # PLIP Features
    python code/models/transmil/train_test_model_fp.py --gpu_id 0 --results_dir 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints' --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/PLIP/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/plip/pt_files' 'results/PLIP/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/plip/pt_files' --label $label --config_json 'code/models/transmil/config/tcgabrca_plip_fts_transmil_config.json' --train_or_test 'train'

    echo "Finished TransMIL (train_model_fp.py) for label: $label"
done

echo 'Finished TransMIL (train_model_fp.py) on TCGA-BRCA Database.'
```



## Testing
Afterward, we can move forward to model testing.

### Using CLAM framework
To test the AM-SB/AM-MB models, you can run the following script:
```bash
#!/bin/bash

echo 'Started CLAM (test_model_fp.py) on TCGA-BRCA Database.'

# List of checkpoint directories for AM_SB (CLAM/ResNet50 Features)
CHECKPOINT_DIRS=('results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/YYYY-MM-DD_hh-mm-ss'\ 
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/YYYY-MM-DD_hh-mm-ss')

 for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Started CLAM (test_model_fp.py) for checkpoint: $checkpoint_dir"
    
    # CLAM/ResNet50 Features
    python code/models/clam/test_model_fp.py --gpu_id 0 --checkpoint_dir $checkpoint_dir --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam/pt_files' 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam/pt_files'

    echo "Finished CLAM (test_model_fp.py) for checkpoint: $checkpoint_dir"
done



# List of checkpoint directories for AM-MB (CLAM/ResNet50 Features)
CHECKPOINT_DIRS=('results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/YYYY-MM-DD_hh-mm-ss')

 for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Started CLAM (test_model_fp.py) for checkpoint: $checkpoint_dir"
    
    # CLAM/ResNet50 Features
    python code/models/clam/test_model_fp.py --gpu_id 0 --checkpoint_dir $checkpoint_dir --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam/pt_files' 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam/pt_files'

    echo "Finished CLAM (test_model_fp.py) for checkpoint: $checkpoint_dir"
done



# List of checkpoint directories for AM-SB and AM-MB (PLIP Features)
 CHECKPOINT_DIRS=('results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/YYYY-MM-DD_hh-mm-ss' \
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/YYYY-MM-DD_hh-mm-ss')


for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Started CLAM (test_model_fp.py) for checkpoint: $checkpoint_dir"
    
    # PLIP Features
    python code/models/clam/test_model_fp.py --gpu_id 0 --checkpoint_dir $checkpoint_dir --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/PLIP/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/plip/pt_files' 'results/PLIP/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/plip/pt_files'

    echo "Finished CLAM (test_model_fp.py) for checkpoint: $checkpoint_dir"
done

echo 'Finished CLAM (test_model_fp.py) on TCGA-BRCA Database.'
```

### Using TransMIL
To test the TransMIL models, you can run the following script:
```bash
#!/bin/bash

echo 'Started TransMIL (train_test_model_fp.py) on TCGA-BRCA Database.'

# List of checkpoint directories for this project (CLAM/ResNet50-based features)
CHECKPOINT_DIRS=('results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/YYYY-MM-DD_hh-mm-ss')

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Started TransMIL (train_test_model_fp.py) for checkpoint directory: $checkpoint_dir"
    
    # CLAM/ResNet50-based features
    python code/models/transmil/train_test_model_fp.py --gpu_id 0 --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam/pt_files' 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam/pt_files' --checkpoint_dir $checkpoint_dir --train_or_test 'test'
    
    echo "Finished TransMIL (train_test_model_fp.py) for checkpoint directory: $checkpoint_dir"
done



 # List of checkpoint directories for this project (PLIP-based features)
CHECKPOINT_DIRS=('results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/YYYY-MM-DD_hh-mm-ss'\
 'results/TransMIL/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/YYYY-MM-DD_hh-mm-ss')

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Started TransMIL (train_test_model_fp.py) for checkpoint directory: $checkpoint_dir"

    # PLIP-based features
    python code/models/transmil/train_test_model_fp.py --gpu_id 0 --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/PLIP/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/plip/pt_files' 'results/PLIP/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/plip/pt_files' --checkpoint_dir $checkpoint_dir --train_or_test 'test'
    
    echo "Finished TransMIL (train_test_model_fp.py) for checkpoint directory: $checkpoint_dir"
done

echo 'Finished TransMIL (train_test_model_fp.py) on TCGA-BRCA Database.'
```



## Heatmap Generation
After training the models, we can proceed to the heatmap generation, to understand their behavior.

### Using CLAM framework and CLAM/ResNet50 features
To generate heatmaps for the CLAM/ResNet50 features, using the CLAM framework, you can run the following script:
```bash
#!/bin/bash

echo 'Started CLAM (create_heatmaps_fp.py) on TCGA-BRCA Database.'

# List of checkpoint directories for AM-SB and AM-MB (CLAM/ResNet50 Features)
CHECKPOINT_DIRS=('results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/YYYY-MM-DD_hh-mm-ss'\ 
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/YYYY-MM-DD_hh-mm-ss'\
 'results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/YYYY-MM-DD_hh-mm-ss')

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Started CLAM (create_heatmaps_fp.py) for checkpoint: $checkpoint_dir"
    
    # CLAM Features
    python code/models/clam/create_heatmaps_fp.py --gpu_id 0 --checkpoint_dir $checkpoint_dir --dataset 'TCGA-BRCA' --base_data_path 'data/TCGA-BRCA' --experimental_strategy 'All' --features_pt_dir 'results/CLAM/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features/clam/pt_files' 'results/CLAM/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features/clam/pt_files' --generate_heatmaps_for 'test' --heatmap_config_file 'code/models/clam/config/tcgabrca_clam_fts_am_sb_heatmap_config.json' --use_histoqc_quality_file 'results/HistoQC/TCGA-BRCA/mmxbrcp/hqc_quality_file.csv' --use_histoqc_seg_masks --verbose

    echo "Finished CLAM (create_heatmaps_fp.py) for checkpoint: $checkpoint_dir"
done

echo 'Finished CLAM (create_heatmaps_fp.py) on TCGA-BRCA Database.'
```

### Using CLAM framework and PLIP features
WORK IN PROGRESS

### Using TransMIL and CLAM/ResNet50 features
WORK IN PROGRESS

### Using TransMIL and PLIP features
WORK IN PROGRESS



# Credits and Acknowledgments
## HistoQC
This [framework](https://github.com/choosehappy/HistoQC) is related to the papers *["HistoQC: An Open-Source Quality Control Tool for Digital Pathology Slides"](http://www.andrewjanowczyk.com/histoqc-an-open-source-quality-control-tool-for-digital-pathology-slides/)* by Janowczyk A., Zuo R., Gilmore H., Feldman M. and Madabhushi A., and *["Assessment of a computerized quantitative quality control tool for kidney whole slide image biopsies"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8392148/)*, Chen Y., Zee J., Smith A., Jayapandian C., Hodgin J., Howell D., Palmer M., Thomas D., Cassol C., Farris A., Perkinson K., Madabhushi A., Barisoni L. and Janowczyk A..

## CLAM
This model and associated [code](https://github.com/mahmoodlab/CLAM) are related to the paper ["Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images"](https://www.nature.com/articles/s41551-020-00682-w) by Ming Y. Lu, Drew F. K. Williamson, Tiffany Y. Chen, Richard J. Chen, Matteo Barbieri and Faisal Mahmood.

## TransMIL
This model and associated [code](https://github.com/szc19990412/TransMIL) are related to the paper "[Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification"](https://proceedings.neurips.cc/paper_files/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html) by Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji and Yongbing Zhang.



# Citation
If you use this repository in your research work, please cite this paper:
```bibtex
@misc{gonçalves2024deep,
      title={{Deep Learning-based Prediction of Breast Cancer Tumor and Immune Phenotypes from Histopathology}}, 
      author={Tiago Gonçalves and Dagoberto Pulido-Arias and Julian Willett and Katharina V. Hoebel and Mason Cleveland and Syed Rakin Ahmed and Elizabeth Gerstner and Jayashree Kalpathy-Cramer and Jaime S. Cardoso and Christopher P. Bridge and Albert E. Kim},
      year={2024},
      eprint={2404.16397},
      archivePrefix={{arXiv}},
      primaryClass={{eess.IV}}
}
```