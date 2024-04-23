# Imports
import os
import argparse
import numpy as np
import time
import pandas as pd

# Project Imports
from utils.tcgabrca_db_utils import get_tcgabrca_folders_and_svs_fpaths
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df



# Function: Generate heatmaps through stitching process
def stitching(file_path, wsi_object, downscale=64, verbose=False):
    
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False, verbose=verbose)
    total_time = time.time() - start
    
    return heatmap, total_time



# Function: Segment a WSI
def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None, use_histoqc_seg_masks=False):

    mask_file_histoqc = (mask_file is not None) and (use_histoqc_seg_masks == True)
    mask_file_only = not(mask_file_histoqc)
    
    # Start Stopwatch
    start_time = time.time()
    
    # Use Segmentation Masks (CLAM or HistoQC)
    if mask_file_only:
        WSI_object.initSegmentation(mask_file)
    elif mask_file_histoqc:
        WSI_object.initHistoQCSegmentation(mask_file)
    
    # Or, perform Segmentation (CLAM)
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    
    # Stop Stopwatch
    seg_time_elapsed = time.time() - start_time   
    

    return WSI_object, seg_time_elapsed



# Function: Generate patches from a WSI
def patching(WSI_object, **kwargs):

    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    patch_time_elapsed = time.time() - start_time

    return file_path, patch_time_elapsed



# Function: Segment and Patch WSI
def seg_and_patch(
        source_dir=None,
        dataset_name=None,
        save_dir=None,
        patch_save_dir=None,
        mask_save_dir=None,
        stitch_save_dir=None,
        patch_size=256,
        step_size=256,
        seg_params={'seg_level':-1, 'sthresh':8, 'mthresh':7, 'close':4, 'use_otsu':False, 'keep_ids':'none', 'exclude_ids':'none'},
        filter_params={'a_t':100, 'a_h':16, 'max_n_holes':8},
        vis_params={'vis_level': -1, 'line_thickness':500},
        patch_params={'use_padding': True, 'contour_fn':'four_pt'},
        patch_level=0,
        use_default_params=False,
        seg=False,
        save_mask=True,
        stitch=False,
        patch=False,
        auto_skip=False,
        process_list=None,
        verbose=False,
        **kwargs):


    # Ensure important conditions to run the code
    assert (source_dir is not None) or ((source_dir is not None) and (dataset_name is not None))

    # Create intermediate variables to facilitate code reading
    source_and_dataset = (source_dir is not None) and (dataset_name is not None)

    # Create conditionals according to the use case
    if source_and_dataset:
        assert dataset_name in ('TCGA-BRCA')
        if dataset_name == 'TCGA-BRCA':
            assert 'experimental_strategy' in kwargs.keys()
            slides = get_tcgabrca_folders_and_svs_fpaths(base_data_path=source_dir, experimental_strategy=kwargs['experimental_strategy'])
            # Example WSI path
            # /autofs/cluster/qtim/datasets/public/TCGA-BRCA/WSI/8f936d42-6deb-43a5-995b-4af18e6a2462/TCGA-A2-A0EY-01Z-00-DX1.2F2428B3-0767-48E0-AC22-443C244CBD16.svs
    
    else:
        slides = sorted(os.listdir(source_dir))
        slides = [slide for slide in slides if os.path.isfile(os.path.join(source_dir, slide))]


    # HistoQC: Quality Assessment of WSI
    hqc_q = None
    if 'use_histoqc_quality_file' in kwargs.keys():
        if kwargs['use_histoqc_quality_file']:
            # Load and read the HistQC quality files
            hqc_q = pd.read_csv(args.use_histoqc_quality_file)

            # Process the dataframe to get the good quality cases
            hqc_q = hqc_q[hqc_q['is_good_quality'] == True]
            hqc_q = hqc_q[['wsi_folder_path', 'is_good_quality']]
            hqc_q = hqc_q['wsi_folder_path']
            hqc_q = list(hqc_q.values)

            # Get the WSI IDs
            hqc_slide_ids = [s.split('/')[-1] for s in hqc_q]

            # Get the dataset WSI IDs
            slide_ids = [s.split('/')[-1] for s in slides]

            # Process the subset of WSI IDs after the results of HistoQC
            slide_ids_ = list()
            for s_id in slide_ids:
                if s_id in hqc_slide_ids:
                    slide_ids_.append(s_id)


            # Build a new list of slides
            slides_ = list()
            for s in slides:
                s_id = s.split('/')[-1]
                if s_id in slide_ids_:
                    slides_.append(s)
            slides = slides_.copy()


    # HistoQC: Use HistoQC Segmentation Masks
    histo_qc_map_masks = None
    if 'use_histoqc_seg_masks' in kwargs.keys():
        if kwargs['use_histoqc_seg_masks']:
            assert seg is False, "The parameter <seg> should be False when using HistoQC Segmentation Masks."
            assert hqc_q is not None, "Please provide the <use_histoqc_quality_file> as a parameter."

            # Build a dictionary that maps the path of the WSI to its HistoQC Segmentation Mask
            histo_qc_map_masks = dict()
            for wsi_path in slides:
                for wsi_m_path in hqc_q:
                    if wsi_path.split('/')[-1] == wsi_m_path.split('/')[-1]:
                        histo_qc_map_masks[wsi_path] = wsi_m_path



    # If we have name of list (.CSV) of images to process with parameters
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)


    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        if verbose:
            print('Detected legacy segmentation CSV file. Legacy support enabled.')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
        'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
        'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
        'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
        'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        
        if verbose:
            print(f"\n\nSlide {i+1} of {total}")
            print(f"Processing {slide}")
        
        # Set the 'process' status to 0
        df.loc[idx, 'process'] = 0
        
        # Get the Slide ID
        slide_id = slide.split('/')[-1]
        slide_id, _ = os.path.splitext(slide_id)


        # If we want to skip slides
        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            if verbose:
                print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue


        # Inialize WSI
        full_path = os.path.join(source_dir, slide)
        WSI_object = WholeSlideImage(full_path)


        # If using default parameters
        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
            
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}


            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0
            
            else:	
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0
            
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
        if w * h > 1e8:
            if verbose:
                print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']



        # Segmentation
        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(
                WSI_object=WSI_object,
                seg_params=current_seg_params,
                filter_params=current_filter_params
            )

        else:
            if histo_qc_map_masks:
                mask_dir = histo_qc_map_masks[slide]
                mask_id = mask_dir.split('/')[-1]
                mask_fname = f"{mask_id}_mask_use.png"
                mask_file = os.path.join(mask_dir, mask_fname)
                WSI_object, seg_time_elapsed = segment(
                    WSI_object=WSI_object,
                    mask_file=mask_file,
                    use_histoqc_seg_masks=True
                )
                if verbose:
                    print(f"HistoQC Segmentation Mask loaded with success from: {mask_file}")

            else:
                pass # TODO


        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)


        # Patching
        patch_time_elapsed = -1 # Default time
        if patch:
            current_patch_params.update(
                {
                    'patch_level': patch_level,
                    'patch_size': patch_size,
                    'step_size': step_size,
                    'save_path': patch_save_dir
                }
            )
            file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)


        # Stitching
        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)


        # Update the status of this slide
        df.loc[idx, 'status'] = 'processed'


        # Update elapsed times
        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed


    # Get the average times
    seg_times /= total
    patch_times /= total
    stitch_times /= total


    # Save results into a .CSV file
    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)

    # Print averaged times per operation
    if verbose:
        print(f"Average Segmentation Time: {seg_times} seconds/slide.")
        print(f"Average Patching Time: {patch_times} seconds/slide.")
        print(f"Average Stiching Time: {stitch_times} seconds/slide.")
        
    return



# Run the script
if __name__ == '__main__':

    # CLI Interface
    parser = argparse.ArgumentParser(description='CLAM: Segment and Patch WSI.')
    parser.add_argument('--source_dir', type=str, required=True, help='path to folder containing raw wsi image files')
    parser.add_argument('--dataset_name', type=str, choices=['TCGA-BRCA'], help="The name of the dataset.")
    parser.add_argument('--tcgabrca_expstr', type=str, choices=['All', 'DiagnosticSlide', 'TissueSlide'], help="The <experimental_strategy> arg for the TCGA-BRCA dataset.")
    parser.add_argument('--step_size', type=int, default=256, help='step_size')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--patch', default=False, action='store_true')
    parser.add_argument('--seg', default=False, action='store_true')
    parser.add_argument('--save_mask', default=False, action='store_true')
    parser.add_argument('--stitch', default=False, action='store_true')
    parser.add_argument('--auto_skip', action='store_true')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save processed data.')
    parser.add_argument('--preset', default=None, type=str, help='predefined profile of default segmentation and filter parameters (.csv)')
    parser.add_argument('--patch_level', type=int, default=0, help='downsample level at which to patch')
    parser.add_argument('--process_list',  type = str, default=None, help='name of list of images to process with parameters (.csv)')
    parser.add_argument('--use_histoqc_quality_file', type=str, default=None, help="Use the quality file generated by the HistoQC framework.")
    parser.add_argument('--use_histoqc_seg_masks', action="store_true", help="Use the segmentation masks generated by the HistoQC framework.")
    parser.add_argument('--verbose', action="store_true", help="Print execution information.")
    args = parser.parse_args()



    # Build the paths of the results' directories
    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    # Create these directories if needed
    for dir_ in [patch_save_dir, mask_save_dir, stitch_save_dir]:
        if not os.path.isdir(dir_):
            os.makedirs(dir_, exist_ok=True)
            # print(f"Directory {dir_} created.")


    # Save experiment arguments
    with open(os.path.join(args.save_dir, "args.txt"), "w") as f:
        f.write(str(args))
    

    # Check if we already have a name of list (.CSV) of images to process with parameters
    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None

    
    # Parameters: Seg
    seg_params = {
        'seg_level':-1,
        'sthresh':8,
        'mthresh':7,
        'close':4,
        'use_otsu':False,
        'keep_ids':'none',
        'exclude_ids':'none'
    }
    with open(os.path.join(args.save_dir, "seg_params.txt"), "w") as f:
        f.write(str(seg_params))


    # Parameters: Filter
    filter_params = {
        'a_t':100,
        'a_h':16,
        'max_n_holes':8
    }
    with open(os.path.join(args.save_dir, "filter_params.txt"), "w") as f:
        f.write(str(filter_params))


    # Parameters: Vis
    vis_params = {
        'vis_level':-1,
        'line_thickness':250
    }
    with open(os.path.join(args.save_dir, "vis_params.txt"), "w") as f:
        f.write(str(vis_params))


    # Parameters: Patch
    patch_params = {
        'use_padding':True,
        'contour_fn':'four_pt'
    }
    with open(os.path.join(args.save_dir, "patch_params.txt"), "w") as f:
        f.write(str(patch_params))


    # If we have preset parameters
    if args.preset:
        preset_df = pd.read_csv(args.preset)
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]


    # Perform segmentation and patching
    seg_and_patch(
        source_dir=args.source_dir,
        dataset_name=args.dataset_name,
        save_dir=args.save_dir,
        patch_save_dir=patch_save_dir,
        mask_save_dir=mask_save_dir,
        stitch_save_dir=stitch_save_dir,
        seg_params=seg_params,
        filter_params=filter_params,
        vis_params=vis_params,
        patch_params=patch_params,
        patch_size=args.patch_size,
        step_size=args.step_size,
        seg=args.seg,
        use_default_params=False,
        save_mask=args.save_mask,
        stitch=args.stitch,
        patch_level=args.patch_level,
        patch=args.patch,
        process_list=process_list,
        auto_skip=args.auto_skip,
        experimental_strategy=args.tcgabrca_expstr,
        use_histoqc_quality_file=args.use_histoqc_quality_file if args.use_histoqc_quality_file else None,
        use_histoqc_seg_masks=args.use_histoqc_seg_masks if args.use_histoqc_seg_masks else None,
        verbose=args.verbose if args.verbose else False
    )
