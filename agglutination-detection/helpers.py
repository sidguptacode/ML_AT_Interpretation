import json
import pandas as pd
import os
import sys
import yaml
from types import SimpleNamespace

def get_position_map():
    """
        Returns a dictionary that maps tray-coordinates (e.g, B2, A3) to (y, x) coordinates.
        Our current code does a rotate and flip to the tray, so this dictionary maps
        coordinates after that rotate and flip.
    """

    # In this definition, A1 is in the top-left corner
    position_map = {}
    for y_ind, y_label in enumerate('ABCDEFGH'):
        for x_ind, x_label in enumerate(range(1, 13)):
            position_map[f"{y_label}{x_label}"] = (y_ind, x_ind)
    return position_map
    
POSITION_MAP = get_position_map()


def compute_fixed_attrs_str(fixed_attr_dict):
    fixed_attr_str = ''
    if fixed_attr_dict != {}:
        fixed_attr_str += '_fixed'
        for attr_name in fixed_attr_dict:
            fixed_attr_str += f"_{attr_name}"
            fixed_attr_str += f"_{fixed_attr_dict[attr_name]}"
    return fixed_attr_str


def create_path(path, directory=False):
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print(f'{dir} does not exist, creating')
        try:
            os.makedirs(dir)
            return True
        except Exception as e:
            print(f'Could not create path {path}')
            raise Exception(e)
    return False


def get_config(experiment_dir, config_file, config_filename='config', load_only=False):
    with open(config_file) as f:
        if not os.path.isfile(config_file):
            print(f"File {config_file} does not exist")
            sys.exit(0)
        data = yaml.load(f, Loader=yaml.FullLoader)

    result = SimpleNamespace(**data)
    if load_only:
        return result

    base_path = experiment_dir
    create_path(base_path, directory=True)

    # save a copy of config for documentation purposes
    with open(f"{base_path}{os.sep}{config_filename}.yml", "w") as f:
        f.write(yaml.dump(data))

    result.experiments_dir = base_path

    return result
