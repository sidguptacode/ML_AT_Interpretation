import sys
import argparse
import os
from os import path
import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from tqdm import trange

sys.path.insert(0,'..')
# Import functions from files
from helpers import get_config, create_path
from experiment_runner import ExperimentRunner
from well_matrix import Well
from constants import MAX_PROCESSES


def main(args):
    """
        Given the command line arguments, create the experiment runner, partition the matrix, and save images.
    """
    config_filepath = f"{args.tray_imgs_dirpath}{os.sep}meta.yml"
    well_metadata = get_config(None, config_filepath, load_only=True)
    sample_opt = SimpleNamespace(max_processes=MAX_PROCESSES, seed=0)
    runner = ExperimentRunner(sample_opt, well_metadata, args.tray_imgs_dirpath, None)
    image_folder_path = args.tray_imgs_dirpath.split(os.sep)
    image_folder_name = image_folder_path[-2] if image_folder_path[-1] == os.sep else image_folder_path[-1]
    runner._create_well_matrix(False, 0, '../_saved_well_matrices', image_folder_name)

    plt.figure(figsize=(8, 8))
    for frame in trange(runner.well_matrix.shape[2]):
        for well_coord in runner.well_matrix.all_selected_well_coords:
            curr_well = runner.well_matrix[well_coord, frame]
            if not isinstance(curr_well, Well):
                continue
            plt.tight_layout()
            plt.axis('off')
            plt.imshow(curr_well.get_image_alias(), cmap='gray')
            plt.tight_layout()
            plt.savefig(f"{args.indiv_wells_dirpath}{os.sep}{image_folder_name}_{well_coord}_{frame}")
            plt.clf()


if __name__ == '__main__':
    """
        Specifies the command line arguments to configure the different experiments.
    """
    parser = argparse.ArgumentParser(description="For segmenting individual wells on LAT trays.")
    parser.add_argument('--tray_imgs_dirpath', type=str, required=True)
    parser.add_argument('--indiv_wells_dirpath', type=str, required=True)
    args = parser.parse_args()
    main(args)

