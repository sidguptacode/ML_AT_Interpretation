import numpy as np
import sys
import argparse
import os
# Import functions from local files
from helpers import get_config
from well_matrix import Well
from feature_extraction.img_proc_fns import IMG_PROC_FNS
from feature_extraction.compute_feature_fns import COMPUTE_FEATURE_FNS, get_feature_ind_dict
from plotting_and_visuals.visualize_well_matrix import plot_wells_list
from constants import KEY_STATES
from experiment_runner import ExperimentRunner

class ExperimentRunnerBlood(ExperimentRunner):
    def run_blood(self, imgs_dict, exterior_masks_dict, visual_output_path, rgb_vis=False) -> None:
        # There is a processing function which requires the original well_np_matrix to be saved. Check for that.
        should_save_key_states = True

        feature_list = [fn_dict['features'] for fn_dict in self.opt.proc_fns]
        feature_list = sum(feature_list, [])
        GET_FEATURE_IND = get_feature_ind_dict(feature_list)

        imgs_feature_vec_dict = {}
        wells = {}

        # Iterate through all the functions we want to process on this well
        for i, fn_dict in enumerate(self.opt.proc_fns):
            fn_name = fn_dict['name']
            if fn_name not in IMG_PROC_FNS.keys():
                raise KeyError("proc_fns in .yml config file does not contain a valid processing function.")
            print(f"Processing {fn_name}")
            for img_url in imgs_dict:
                img = imgs_dict[img_url]
                curr_well = wells.get(img_url, None)
                if curr_well is None:
                    curr_well = Well(img, self.opt.feature_vec_size, type='BLOOD')
                    exterior_mask = exterior_masks_dict[img_url]
                    curr_well.exterior_mask = exterior_mask
                    curr_well.well_saved_states['exterior_mask'] = exterior_mask
                    wells[img_url] = curr_well
                # Process the function on this well
                extra_args = curr_well.well_saved_states if 'mask_binary' in fn_name else fn_dict['hyperparams']
                IMG_PROC_FNS[fn_name](curr_well.get_image_alias(), extra_args)
                if fn_name in KEY_STATES and should_save_key_states:
                    curr_well.well_saved_states[fn_name] = curr_well.get_image_alias().copy()
                # Iterate through all the features we want to compute on this well, after processing this function
                for feature in fn_dict['features']:
                    # Compute the feature, and store it in this well's feature vector
                    feature_val = COMPUTE_FEATURE_FNS[feature](curr_well)
                    feature_ind = GET_FEATURE_IND[feature]
                    curr_well.update_feature_vec(feature_val, feature_ind)
                imgs_feature_vec_dict[img_url] = curr_well.feature_vec
            # Visualize the well matrix at this step, if it's specified.
            if i in self.opt.plot_well_matrix['steps']:
                print(f"Visualizing {fn_name}")
                plot_wells_list(wells, fn_name, visual_output_path, rgb=rgb_vis)
            print(f"Finished processing {fn_name}")
        return imgs_feature_vec_dict
