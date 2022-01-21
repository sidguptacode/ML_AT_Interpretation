
import sys
sys.path.append('..')
from experiment_runner import ExperimentRunner
from well_matrix import WellMatrix, Well
import os
import argparse
from helpers import get_config, create_path
import numpy as np 

def main(args):
    MAX_SEQ_LEN = 300
    dataset = []
    curr_seq_len = -1
    seq_lens = []
    trays_completed = 0
    dirs_to_skip = ['20210728_22_34', '.DS_Store']
    imgdirs = os.listdir(args.full_imgs_dir)
    # imgdirs.sort()
    # imgdirs.reverse()
    # print(imgdirs)
    for imgdir in imgdirs:
    # for imgdir in ['2021-04-30_rabbit']:
        if imgdir in dirs_to_skip:
            continue
        imgdir = f"{args.full_imgs_dir}/{imgdir}"
        metadata_config_path = f"{imgdir}{os.sep}meta.yml"
        if not os.path.isfile(metadata_config_path):
            continue
        print(f"Running {imgdir}")
        opt = get_config(args.experiment_dir, args.config_file)
        well_metadata = get_config(args.experiment_dir, metadata_config_path, config_filename='meta')
        runner = ExperimentRunner(opt, well_metadata, imgdir, 'ngboost')
        runner.run(args.experiment_dir, resave=args.resave, plot_smoothed = False, plot_ngboost=args.plot_ngboost=='true')
        for group in runner.well_matrix.groups:
            mapped_df_dists = runner.well_matrix.main_df_subgrouped[group]['df'].applymap(lambda well: well.agg_score_dist if isinstance(well, Well) else -1)
            concs = [conc for conc in mapped_df_dists.columns if conc != 'blank']
            concs = np.unique(np.array(concs))
            for conc in concs:
                dists = mapped_df_dists[conc].values.T
                for i in range(len(dists)):
                    well_dists = [dist for dist in dists[i] if dist != -1]
                    well_agg_score_dist_params = [[well_dist.params['s'], well_dist.params['scale']] for well_dist in well_dists]
                    curr_seq_len = len(well_agg_score_dist_params)
                    np_dist_params = np.zeros((MAX_SEQ_LEN, 2))
                    np_dist_params[0:len(well_agg_score_dist_params), :] = np.array(well_agg_score_dist_params)
                    dataset.append(np_dist_params)
        seq_lens.append(curr_seq_len)
        print(np.array(dataset).shape)
        print(np.array(seq_lens).shape)
        print(seq_lens[-1])
        print(i)
        print('====')
        trays_completed += 1
        if trays_completed % 2 == 0:
            dataset_save = np.array(dataset)
            seq_lens_save = np.array(seq_lens)
            print(dataset_save.shape)
            np.savez(f'dataset_dhmm_{trays_completed}.npz', X=dataset_save, seq_lens=seq_lens_save)
            print("Dataset saved!")

    dataset = np.array(dataset)
    seq_lens = np.array(seq_lens)
    print(dataset.shape)
    np.savez(f'dataset_dhmm.npz', X=dataset, seq_lens=seq_lens)
    print("Dataset saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates a dataset of sequences for a DHMM.")
    parser.add_argument('--full_imgs_dir', type=str, required=True)
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--resave', type=str, required=True)
    parser.add_argument('--plot_ngboost', type=str, required=False, help="If set to true, then it will generate plots of each well, with the labels vs. predictions, smoothed.")
    args = parser.parse_args()
    main(args)