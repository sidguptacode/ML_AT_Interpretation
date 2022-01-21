from __future__ import print_function
import os.path
import argparse
import numpy as np
import sys
from types import SimpleNamespace
sys.path.insert(0,'..')
from download_owncloud_tray_imgs import download_from_oc
from helpers import get_config
from experiment_runner import ExperimentRunner
from training_ml.pull_labels import pull_labels
from well_matrix import Well

def main(args):
    opt = get_config(None, args.create_dataset_config, load_only=True)
    opt.run_type = 'create_dataset'

    # Get the labels from the spreadsheet
    labels_dict = pull_labels(False)

    # Download the trays_labelled images, if they're not already downloaded
    tray_names = list(labels_dict.keys())
    download_from_oc(args.full_tray_imgs_dir, num_trays_to_download=len(tray_names), trays_to_download=tray_names)

    print(f"Annotated trays loading: {len(tray_names)}")

    # The contents of the csv file
    csv_contents = ""

    for tray_name in tray_names:
        print(f"Processing {tray_name}")
        labels = labels_dict[tray_name]
        tray_folder_dirpath = f"{args.full_tray_imgs_dir}{os.sep}{tray_name}"
        config_filepath = f"{tray_folder_dirpath}{os.sep}meta.yml"
        well_metadata = get_config(None, config_filepath, load_only=True)
        runner = ExperimentRunner(opt, well_metadata, tray_folder_dirpath)
        runner._create_well_matrix(should_save_key_states=True, feature_vec_size=opt.feature_vec_size, saved_well_matrices_path='../_saved_well_matrices', matrix_filename=tray_name, labels=labels, resave=args.resave == 'true')

        group_label_counts = {group: np.zeros(5) for group in runner.well_matrix.groups}
        coord_to_group_dict = {}
        for group in runner.well_matrix.groups:
            # Get the coordinates of all Wells in this group
            group_coords = runner.well_matrix.main_df_subgrouped[group]['maps'].keys()
            # Map each coordinate to a group
            for coord in group_coords:
                coord_to_group_dict[coord] = group

        label_counts = np.zeros(5)
        # Append the feature vectors and labels to the dataset
        for well_coord, frame in labels:
            curr_well = runner.well_matrix[well_coord, frame]
            if not isinstance(curr_well, Well):
                continue
            if curr_well.label != -1 and labels[well_coord, frame] == curr_well.label:
                label_val = int(labels[well_coord, frame])
                label_counts[label_val] += 1
                well_group = coord_to_group_dict[well_coord]
                group_label_counts[well_group][label_val] += 1
        
        if args.dists_to_print == 'trays':
            csv_str = ""
            csv_str += f"{tray_name},"
            csv_str += f"{len(runner.well_matrix.all_selected_well_coords)},"
            csv_str += f"{runner.well_matrix.shape[2]},"
            csv_str += f"{runner.well_matrix.shape[2] * len(runner.well_matrix.all_selected_well_coords)},"
            csv_str += f"{sum(label_counts)},"
            label_props = label_counts / sum(label_counts)
            for i in range(5):
                csv_str += "{:.2f},".format(label_props[i])
            for i in range(5):
                csv_str += "{:.2f},".format(label_counts[i])
            csv_contents += csv_str[:-1] + '\n'

        if args.dists_to_print == 'groups':
            for group in runner.well_matrix.groups:
                csv_str = ""
                csv_str += f"{tray_name}_{group},"
                all_coords_in_group = 0
                for group_ in coord_to_group_dict.values():
                    if group_ == group:
                        all_coords_in_group += 1
                csv_str += f"{all_coords_in_group},"
                csv_str += f"{runner.well_matrix.shape[2]},"
                csv_str += f"{runner.well_matrix.shape[2] * all_coords_in_group},"
                label_counts = group_label_counts[group]
                csv_str += f"{sum(label_counts)},"
                label_props = label_counts / sum(label_counts)
                for i in range(5):
                    csv_str += "{:.2f},".format(label_props[i])
                for i in range(5):
                    csv_str += "{:.2f},".format(label_counts[i])
                csv_contents += csv_str[:-1] + '\n'

    with open(f"./{args.dists_to_print}_distributions.csv", "w") as outf:
        outf.write(csv_contents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pulls labels from the spreadsheet, downloads labelled trays from ownCloud")
    parser.add_argument('--full_tray_imgs_dir', type=str, required=True)
    parser.add_argument('--create_dataset_config', type=str, required=True, help="NOT the meta.yml file, \
            but rather the ml_test_config.yml file used by the experiment_runner")
    parser.add_argument('--dists_to_print', type=str, required=True, help="The kind of distribution you want to print (`trays` for wells in trays, `groups`  \
            for wells in groups")
    parser.add_argument('--resave', type=str, required=True, help="Whether or not we should re-compute the WellMatrix objects, \
            even if they have already been computed and saved.")
    args = parser.parse_args()
    main(args)