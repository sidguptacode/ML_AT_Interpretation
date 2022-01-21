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


def pull_high_disagreements(labels_dict, labels_dict_all, dataset_name, opt, args):
    # Download the trays_labelled images, if they're not already downloaded
    tray_names = list(labels_dict.keys())
    download_from_oc(args.full_tray_imgs_dir, num_trays_to_download=len(tray_names), trays_to_download=tray_names)

    print(f"Annotated trays loading: {len(tray_names)}")

    gitlab_prefix = 'https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/'
    with open(f"./high_disagreements.txt", "w") as outf:

        for tray_name in tray_names:
            print(f"Processing {tray_name}")
            labels = labels_dict[tray_name]
            all_labels = labels_dict_all[tray_name]
            # Append the feature vectors and labels to the dataset
            for well_coord, frame in labels:
                labels_lst = np.array(all_labels[well_coord, frame])
                agreement_amt = (np.count_nonzero(labels_lst == np.bincount(labels_lst).argmax()) / len(labels_lst))
                # See if more than half of these labels disagree
                if (agreement_amt <= 0.5):
                    outf.write(f"{gitlab_prefix}{tray_name}_{well_coord}_{frame}.png,{labels_lst}\n")  


def main(args):
    opt = get_config(None, args.create_dataset_config, load_only=True)
    opt.run_type = 'create_dataset'
    # Get the labels from the spreadsheet
    _, multi_labels_dict, multi_labels_dict_all = pull_labels(False)
    pull_high_disagreements(multi_labels_dict, multi_labels_dict_all, "dataset", opt, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pulls labels from the spreadsheet, downloads labelled trays from ownCloud")
    parser.add_argument('--full_tray_imgs_dir', type=str, required=True)
    parser.add_argument('--create_dataset_config', type=str, required=True, help="NOT the meta.yml file, \
            but rather the ml_test_config.yml file used by the experiment_runner")
    args = parser.parse_args()
    main(args)