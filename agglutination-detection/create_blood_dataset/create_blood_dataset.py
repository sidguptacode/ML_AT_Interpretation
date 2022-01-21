from __future__ import print_function
import argparse
import numpy as np
import sys
sys.path.insert(0,'..')
from training_ml.training_helpers import get_normalization_constants
from experiment_runner import ExperimentRunner
from helpers import get_config
from training_ml.pull_labels import pull_labels
from core_process import compute_imgs_dict
import os


def main(args):

    imgs_dict, exterior_masks_dict = compute_imgs_dict(args.load_blood_imgs == 'true')

    labels_dict = pull_labels(False, 'bloodversion2')
    url_names = list(labels_dict.keys())

    # Process these trays, assign the wells labels, and build a dataset for training
    X, y = [], []
    imgs = []
    X_urls = []
    groups = []
    gitlab_prefix = 'https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/'

    # Get a dictionary of feature vectors from the dictionary of images
    opt = get_config(None, f"..{os.sep}..{os.sep}configs{os.sep}ml_test_config_blood.yml", load_only=True)
    opt.run_type = 'create_dataset'
    runner = ExperimentRunner(opt, None, None)
    imgs_feature_vec_dict = runner.run_blood(imgs_dict, exterior_masks_dict, f'.{os.sep}blood_visuals', rgb_vis=False)

    for url_name in url_names:
        label_ = labels_dict[url_name]
        imgs.append(imgs_dict[url_name])
        y.append(label_)
        X_urls.append(f"{gitlab_prefix}{url_name}")
        X.append(imgs_feature_vec_dict[url_name])
        groups.append(url_name)

    # Normalize the dataset
    X, y, groups, imgs = np.array(X), np.array(y), np.array(groups), np.array(imgs)

    X_urls = np.array(X_urls)
    print(f"Number of data samples: {len(imgs)}")
    mean_x, var_x = get_normalization_constants(X)
    X -= mean_x
    X /= var_x

    # Save the dataset
    np.savez(f"{args.dataset_outpath}/dataset_blood.npz", X=X, y=y, X_urls=X_urls, groups=groups, imgs=imgs)
    np.savez(f"{args.dataset_outpath}/ml_normalization_constants.npz", mean_x=mean_x, var_x=var_x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pulls labels from the spreadsheet, and processes blood images")
    parser.add_argument('--load_blood_imgs', type=str, required=True, help="If the processed blood images are saved, they will be loaded instead of computed")
    parser.add_argument('--create_dataset_config', type=str, required=True, help="NOT the meta.yml file, \
            but rather the ml_test_config.yml file used by the experiment_runner")
    parser.add_argument('--dataset_outpath', type=str, required=True, help="The path to which the created dataset is saved.")
    args = parser.parse_args()
    main(args)