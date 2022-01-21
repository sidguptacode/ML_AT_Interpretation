import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from itertools import product
import pandas as pd
from well_matrix import Well
import datetime
from datetime import datetime
import os
from sklearn.cluster import KMeans
from collections import Counter

from well_matrix_creation.preprocess_tray import rotate_n_flip, cor_lum, crop_to_plate, find_wells, read_wells, map_columns
from well_matrix_creation.find_clusters import find_clusters
from well_matrix_creation.subgrouper import Subgrouper

class PreprocessedTray:
    def __init__(self, path, well_feature_vec_size, should_save_key_states_well=True, img_rot=None):
        self.path = path
        self.img_rot = img_rot
        self.date_obj = None
        self.plate_rot = None
        self.plate_cor = None
        self.circles = None
        self.well_images = None
        self.mapper = None
        self.assigned_map = None
        self.well_feature_vec_size = well_feature_vec_size
        self.should_save_key_states_well = should_save_key_states_well

        self.img = cv2.imread(self.path)
        self.name_to_date()

    def name_to_date(self):
        self.date_obj = datetime.strptime(self.path.split(os.sep)[-1], '%Y%m%d_%H%M%S.jpg')

    def process(self):
        plate = crop_to_plate(self.img)
        if self.img_rot is None:
            raise Exception("Image rotation not defined")

        if self.img_rot:
            self.plate_rot = rotate_n_flip(plate, rot=cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            self.plate_rot = rotate_n_flip(plate)

        self.plate_cor = cor_lum(self.plate_rot)

        gray_plate_cor = cv2.cvtColor(self.plate_cor, cv2.COLOR_BGR2GRAY)
        self.circles = find_wells(gray_plate_cor)
        # self.mapper contains the (x, y) coordinates of the wells in this tray.
        self.well_images, self.mapper = read_wells(gray_plate_cor, self.circles)
        self.well_objects = [Well(well_image, self.well_feature_vec_size, type='COVID') for well_image in self.well_images]

    def to_frame(self):
        if self.well_images is not None:
            # each well here is a flattened image vector, which gets represented as a column in the df.
            df = pd.DataFrame({tuple(map_): [obj] for map_, obj in zip(self.mapper, self.well_objects)})

            return df


def multi_analysis(path, well_feature_vec_size, should_save_key_states_well, img_rot):
    """
        Allows us to process a tray at the provided path.
    """
    wells = PreprocessedTray(path, well_feature_vec_size, should_save_key_states_well, img_rot=img_rot)
    wells.process()
    return wells


def multi_analysis_wrap(args):
    """
        Wrapper for multiprocessing.
    """
    return multi_analysis(*args)


def compute_well_matrix_dfs(folder_path, well_feature_vec_size, should_save_key_states_well, well_metadata, max_processes):
    """
        Reads the well image dataset, and stores all well images in a 5D numpy array.
    """
    if folder_path[-1] != '/':
        folder_path += '/'

    # Iterates through every image in the folder
    img_file_list = glob(folder_path+'*.jpg')
    well_trays = []
    print('Preprocessing tray image')
    with Pool(processes=max_processes) as p:
        with tqdm(total=len(img_file_list)) as pbar:
            for i, well_tray in enumerate(p.imap_unordered(multi_analysis_wrap,
                                                        product(img_file_list, [well_feature_vec_size], [should_save_key_states_well], [well_metadata.A1_top]))):
                well_trays.append(well_tray)
                pbar.update()

    print('Finding clusters')
    main_df = find_clusters(well_trays)
    main_df.columns = map_columns(main_df)

    # Drop all wells that aren't being analyzed
    wells_analyzed = well_metadata.sample_wells
    main_df.drop(main_df.columns.difference(wells_analyzed), 1, inplace=True)

    return main_df


def subgroup_main_df(folder_path, main_df, group_data):
    subgrouper = Subgrouper(main_df, group_data, folder_path)
    main_df_subgrouped = subgrouper.map_all()

    return main_df_subgrouped
    