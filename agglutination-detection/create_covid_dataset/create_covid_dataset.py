from __future__ import print_function
import argparse
import numpy as np
import sys
import os
sys.path.insert(0,'..')
from training_ml.training_helpers import get_normalization_constants
from experiment_runner import ExperimentRunner
from download_owncloud_tray_imgs import download_from_oc
from helpers import get_config
from training_ml.pull_labels import pull_labels
from well_matrix import Well
import matplotlib.pyplot as plt


def run_experiment(labels, tray_name, args, opt):
    tray_folder_dirpath = f"{args.full_tray_imgs_dir}{os.sep}{tray_name}"
    config_filepath = f"{tray_folder_dirpath}{os.sep}meta.yml"
    well_metadata = get_config(None, config_filepath, load_only=True)
    runner = ExperimentRunner(opt, well_metadata, tray_folder_dirpath)
    runner._create_well_matrix(should_save_key_states=True, feature_vec_size=opt.feature_vec_size, saved_well_matrices_path='../_saved_well_matrices', matrix_filename=tray_name, labels=labels, resave=args.resave == 'true')
    runner.run(f"{tray_folder_dirpath}/experiments", saved_well_matrices_path='../_saved_well_matrices')
    return runner


def create_and_save_dataset(labels_dict, opt, args, unilabel_trays=None):
    # Download the trays_labelled images, if they're not already downloaded
    tray_names = list(labels_dict.keys())
    download_from_oc(args.full_tray_imgs_dir, num_trays_to_download=len(tray_names), trays_to_download=tray_names)

    print(f"Annotated trays loading: {len(tray_names)}")

    # Process these trays, assign the wells labels, and build a dataset for training
    X, y = [], []
    imgs = []
    X_urls = []
    # Groups represent "groupings" of the data, which will be preserved when dividing the dataset into cross-validation folds.
    groups = []
    i = 0
    gitlab_prefix = 'https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/'
    for tray_name in tray_names:
        # if unilabel_trays != None: # TODO: We're only using my unilabels for the recent two trays
        #     if tray_name not in ['20210708_14_38', '20210708_13_33']:
        #         continue
        print(f"Processing {tray_name}")
        labels = labels_dict[tray_name]
        runner = run_experiment(labels, tray_name, args, opt)
        # Append the feature vectors and labels to the dataset
        for well_coord, frame in labels:
            i += 1
            curr_well = runner.well_matrix[well_coord, frame]
            if not isinstance(curr_well, Well):
                continue
            if curr_well.label != -1 and labels[well_coord, frame] == curr_well.label:
                groups.append(f"{tray_name}_{well_coord}")
                X.append(curr_well.feature_vec)
                X_urls.append(f"{gitlab_prefix}{tray_name}_{well_coord}_{frame}.png")
                y.append(labels[well_coord, frame])
                imgs.append(curr_well.well_saved_states['original'])

    # Normalize the dataset
    X, y, groups, imgs = np.array(X), np.array(y), np.array(groups), np.array(imgs)
    X_urls = np.array(X_urls)    
    dataset = {"X": X, "y": y, "X_urls": X_urls, "groups": groups, "imgs": imgs}
    return dataset


def merge_datasets(dataset1, dataset2):
    new_dataset = {}
    for key in dataset1:
        new_dataset[key] = np.concatenate((dataset1[key], dataset2[key]), axis=0)
    return new_dataset


def save_dataset(dataset, outpath, dataset_name):
    # Save the dataset
    X = dataset["X"]
    print(f"Number of data samples: {len(X)}")
    mean_x, var_x = get_normalization_constants(X)
    X -= mean_x
    X /= var_x
    np.savez(f'{outpath}/{dataset_name}.npz', X=dataset["X"], y=dataset["y"], X_urls=dataset["X_urls"], groups=dataset["groups"], imgs=dataset["imgs"])
    np.savez('../ml_normalization_constants.npz', mean_x=mean_x, var_x=var_x)


def main(args):
    opt = get_config(None, args.create_dataset_config, load_only=True)
    opt.run_type = 'create_dataset'
    # Get the labels from the spreadsheet
    uni_labels_dict, multi_labels_dict, _ = pull_labels(False)
    print("done pull_labels!")
    multi_labels_dataset = create_and_save_dataset(multi_labels_dict, opt, args)

    if args.joined == 'true':
        uni_labels_dataset = create_and_save_dataset(uni_labels_dict, opt, args)
        final_dataset = merge_datasets(multi_labels_dataset, uni_labels_dataset)
        print(f"Length of final_dataset['X']: {len(final_dataset['X'])}")
    else:
        final_dataset = multi_labels_dataset
    save_dataset(final_dataset, args.dataset_outpath, "dataset")

    if args.warmstart == 'true':
        uni_dataset = create_and_save_dataset(uni_labels_dict, opt, args)
        save_dataset(uni_dataset, args.dataset_outpath, "dataset_warm")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pulls labels from the spreadsheet, downloads labelled trays from ownCloud")
    parser.add_argument('--full_tray_imgs_dir', type=str, required=True)
    parser.add_argument('--create_dataset_config', type=str, required=True, help="NOT the meta.yml file, \
            but rather the ml_test_config.yml file used by the experiment_runner")
    parser.add_argument('--resave', type=str, required=True, help="Whether or not we should re-compute the WellMatrix objects, \
            even if they have already been computed and saved.")
    parser.add_argument('--dataset_outpath', type=str, required=True, help="The path to which the created dataset is saved.")
    parser.add_argument('--warmstart', type=str, required=False, help="If this equals true, then we will also create a warmstart dataset using the unilabels.")
    parser.add_argument('--joined', type=str, required=False, help="If this equals true, then we will join the unilabel and multilabel datasets.")
    args = parser.parse_args()
    main(args)