import os

from helpers import get_config
from experiment_runner import ExperimentRunner

experiment_folder = '20210720_13_26'

experiment_dir = f'../{experiment_folder}/experiments'
img_folder = f'../{experiment_folder}'
config_file = f'./configs/ml_test_config.yml'
ml_model_type = 'ngboost'  # 'regressor'
metadata_config_path = f"{img_folder}{os.sep}meta.yml"

if __name__ == '__main__':
    opt = get_config(experiment_dir, config_file)
    well_metadata = get_config(experiment_dir, metadata_config_path, config_filename='meta')
    runner = ExperimentRunner(opt, well_metadata, img_folder, ml_model_type)
    runner.run(experiment_dir)
