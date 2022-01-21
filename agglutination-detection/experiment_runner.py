import numpy as np
import sys
import argparse
from os import path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import joblib
import pandas as pd
from tqdm import trange
import math
import os
# Import functions from local files
from helpers import get_config, create_path
from well_matrix_creation.compute_well_matrix import compute_well_matrix_dfs, subgroup_main_df
from well_matrix import WellMatrix, Well
from feature_extraction.img_proc_fns import IMG_PROC_FNS
from feature_extraction.compute_feature_fns import COMPUTE_FEATURE_FNS, get_feature_ind_dict
from plotting_and_visuals.visualize_well_matrix import plot_well_matrix_over_time
from plotting_and_visuals.plot_well_matrix import get_stats, plot_scores, plot_table, plot_smoothed_scores, \
    wells_for_csv, plot_wells, ngboost_interval_plot
from testing_ml.ml_predict import ml_predict, add_ground_truth
from constants import KEY_STATES


class ExperimentRunner():
    """
        Defines an ExperimentRunner instance, which manages how a WellMatrix
        is read and preprocessed.

        Attributes:
            - self.opt: Options that are read from a .yml config file.
            - self.well_matrix: An object representing a WellMatrix, of size (tray_width, tray_height, num_frames).
            - self.well_metadata: Options that are loaded from a meta.yml file in the image dataset. Specifically,
                they specify the positions of selected wells, and group configurations.
            - self.img_folder: The path to the original image data folder
            - self.ml_model_type: The type of ML model being used for testing-- one of (regressor, classifier, ngboost)
    """

    def __init__(self, opt: dict, well_metadata: dict, img_folder: str, ml_model_type='regressor') -> None:
        """
        Creates an ExperimentRunner instance, and sets the options
        that are read by the .yml config file.
      """
        if opt.seed is None:
            raise ValueError(
                "The seed was not specified correctly in the config file")
        np.random.seed(opt.seed)
        self.opt = opt
        self.img_folder = img_folder
        self.well_metadata = well_metadata
        self.seed = opt.seed
        self.ml_model_type = ml_model_type

    def _label_wells(self, main_df, labels):
        """
            Used for visualization/debug purposes only, so we can plot the ground truth values of the wells.
        """
        for well_coord, frame in labels:
            # Assign the label, if we can
            wells_at_frame = main_df.loc[(slice(None), slice(None), slice(None), int(frame)), :]
            curr_well = wells_at_frame[well_coord].values[0]
            if not isinstance(curr_well, Well):
                continue
            curr_well.label = labels[well_coord, frame]

    def _create_well_matrix(self, should_save_key_states: bool, feature_vec_size: int, saved_well_matrices_path: str,
                            matrix_filename: str, labels=None, resave=False) -> None:
        """
            Transforms the LAT tray into a WellMatrix that stores each Well.
        """
        dataframes_path = f"{saved_well_matrices_path}/{matrix_filename}"
        if path.exists(f"{dataframes_path}.h5") and not resave:
            main_df = pd.read_hdf(f'{dataframes_path}.h5', key='main_df')
            main_df.labelled = labels is not None
            print("Finished loading well matrix")
        else:
            print(f"Computing well_matrix and saving in {saved_well_matrices_path}")
            main_df = compute_well_matrix_dfs(self.img_folder, feature_vec_size, should_save_key_states,
                                              self.well_metadata, self.opt.max_processes)
            main_df.labelled = labels is not None
            if main_df.labelled:
                # Label the Well objects inside the main_df
                self._label_wells(main_df, labels)
            create_path(saved_well_matrices_path, directory=True)
            main_df.to_hdf(f'{dataframes_path}.h5', key='main_df')
        main_df_subgrouped = subgroup_main_df(self.img_folder, main_df, self.well_metadata.groups)
        self.well_matrix = WellMatrix(main_df, main_df_subgrouped)
        print('Finished creating well matrix')

    def run(self, experiment_dir, saved_well_matrices_path='./_saved_well_matrices',
            resave=False, plot_smoothed=False, plot_ngboost=False) -> None:
        """
            Runs the experiment specified by the provided config file.
        """
        # There is a processing function which requires the original well_np_matrix to be saved. Check for that.
        should_save_key_states = False
        for proc_fn_dict in self.opt.proc_fns:
            if 'mask_binary' in proc_fn_dict['name']:
                should_save_key_states = True
                break

        # Get the list of features which are to be processed into a feature vector.
        feature_list = [fn_dict['features'] for fn_dict in self.opt.proc_fns]
        feature_list = sum(feature_list, [])
        GET_FEATURE_IND = get_feature_ind_dict(feature_list)

        # Create the WellMatrix object.
        image_folder_path = self.img_folder.split("/")
        image_folder_name = image_folder_path[-2] if self.img_folder[-1] == '/' else image_folder_path[-1]
        self._create_well_matrix(should_save_key_states, self.opt.feature_vec_size, saved_well_matrices_path,
                                 image_folder_name, resave=resave == 'true')

        # Plot the WellMatrix at various time-steps, if that is specified in the config .yml file.
        if self.opt.plot_well_matrix['plot_wells_at_times'] != []:
            print("Plotting original wells at various timepoints")
            for group in self.well_matrix.groups:
                plot_wells(self.well_matrix.main_df_subgrouped, group, experiment_dir + "/",
                           self.opt.plot_well_matrix['plot_wells_at_times'], show_plots=None, save_plots=True)

        # Iterate through all the functions we want to process on this well
        for i, fn_dict in enumerate(self.opt.proc_fns):
            fn_name = fn_dict['name']
            if fn_name not in IMG_PROC_FNS.keys():
                raise KeyError("proc_fns in .yml config file does not contain a valid processing function.")
            print(f"Processing {fn_name}")
            # Iterate through all selected wells in the image dataset
            for frame in trange(self.well_matrix.shape[2]):
                for well_coord in self.well_matrix.all_selected_well_coords:
                    curr_well = self.well_matrix[well_coord, frame]
                    if not isinstance(curr_well, Well):
                        # If this well was not detected by OpenCV, skip it's processing
                        assert math.isnan(curr_well)
                        continue
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
            # Visualize the well matrix at this step, if it's specified. (for DEBUG purposes only!)
            if i in self.opt.plot_well_matrix['steps']:
                print(f"Visualizing {fn_name}")
                create_path(experiment_dir, directory=True)
                rgb = 'draw_contours' in fn_dict['features']
                plot_well_matrix_over_time(self.well_matrix, experiment_dir, image_folder_name, fn_name, rgb=rgb)
            print(f"Finished processing {fn_name}")
        if self.opt.run_type == 'create_dataset':
            return
        elif self.opt.run_type == 'ml_test':
            # Load the normalization constants from the dataset
            normalization_constants = np.load('./ml_normalization_constants.npz')
            mean_x, var_x = normalization_constants['mean_x'], normalization_constants['var_x']
            ml_predict(self.well_matrix, self.ml_model_type, mean_x, var_x)
            add_ground_truth(self.well_matrix)
        else:
            raise Exception("Could not identify run_type in the config.")

        # Plotting various features (RSD, ml_agglution_score, total_area_of_all_blobs)
        if self.opt.plots != []:
            csv_export = {}
            for i, plot_feature in enumerate(self.opt.plots):
                for group in self.well_matrix.groups:
                    if csv_export.get(group, None) is None:
                        csv_export[group] = {}
                    fig = plt.figure(figsize=(10, 5))
                    gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
                    axes = []
                    axes.append(fig.add_subplot(gs[0, :-1]))
                    axes.append(fig.add_subplot(gs[:, -1]))
                    mpl_table = plot_table(self.well_matrix.main_df_subgrouped, group, axes[-1])
                    feature_ind = GET_FEATURE_IND[plot_feature]
                    mapped_df_feature = self.well_matrix.main_df_subgrouped[group]['df'].applymap(
                        lambda well: well.feature_vec[feature_ind] if isinstance(well, Well) else -1)
                    mean, std, count = get_stats(mapped_df_feature)
                    plot_scores(self.well_matrix.main_df_subgrouped, mean, std, count, group, experiment_dir,
                                plot_feature, fig, axes[0], mpl_table)
                    csv_export[group][plot_feature] = wells_for_csv(mapped_df_feature,
                                                                    self.well_matrix.main_df_subgrouped[group][
                                                                        'meta_table'], experiment_dir)
                    fig.savefig(experiment_dir + f'/group_{group}_analysis_{plot_feature}.svg')

        if plot_ngboost == 'true' and self.ml_model_type == 'ngboost':
            print("Plotting NGBoost intervals")
            ngboost_interval_plot(self.well_matrix, image_folder_name, experiment_dir, GET_FEATURE_IND)

        # If we are plotting the ground truths, let's also plot it compared to predictions
        if plot_smoothed == 'true' and "ground_truth" in self.opt.plots:
            print("Plotting Smoothed scores")
            plot_smoothed_scores(self.well_matrix, image_folder_name, experiment_dir, GET_FEATURE_IND)

        # We save the scores in a .csv to investigate post-processing
        csv_export = {e: pd.concat(csv_export[e], axis=1) for e in csv_export}
        csv_export = pd.concat(csv_export)
        csv_export.to_csv(experiment_dir + '/data.csv')


def main(args):
    """
        Given the command line arguments, run the experiments and plot results.
    """
    opt = get_config(args.experiment_dir, args.config_file)
    metadata_config_path = f"{args.img_folder}{os.sep}meta.yml"
    well_metadata = get_config(args.experiment_dir, metadata_config_path, config_filename='meta')
    runner = ExperimentRunner(opt, well_metadata, args.img_folder, args.ml_model_type)
    runner.run(args.experiment_dir, resave=args.resave, plot_smoothed=args.plot_smoothed == 'true',
               plot_ngboost=args.plot_ngboost == 'true')


if __name__ == '__main__':
    """
        Specifies the command line arguments to configure the different experiments.
    """
    parser = argparse.ArgumentParser(
        description="For running different kinds of feature-extraction experiments on LAT trays.")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--ml_model_type', type=str, required=False,
                        help="The type of model (either classifier or regressor or none)")
    parser.add_argument('--resave', type=str, required=False, help="Whether or not we should resave the datasets")
    parser.add_argument('--plot_smoothed', type=str, required=False,
                        help="If set to true, then it will generate plots of each well, with the labels vs. predictions, smoothed.")
    parser.add_argument('--plot_ngboost', type=str, required=False,
                        help="If set to true, then it will generate a distrubtion plot of each well.")
    args = parser.parse_args()
    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    main(args)
