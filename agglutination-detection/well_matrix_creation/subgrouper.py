import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from datetime import datetime
import os
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
from itertools import product
from tqdm import tqdm
from collections import Counter


class Subgrouper:
    def __init__(self, df, meta, folder,
                 sub_zero=False, show_plots=True, save_plots=False):
        self.df = df
        self.meta = meta
        self.folder = folder
        self.sub_zero = sub_zero
        self.show_plots = show_plots
        self.save_plots = save_plots

        print('Looking for subgroups...')
        self.sub_groups = self.find_subgroups()
        print('Creating DataFrame...')
        # TODO: Might still want to keep the rsd metric, just in case.
        # self.df_rsd = self.get_rsd()
        # print('Mapping groups to meta...')
        # self.stats = self.map_all()

    def find_subgroups(self):
        groups = pd.DataFrame(self.meta)
        attributes = groups.index
        unique = [id_ for i, id_ in enumerate(attributes) if
                  id_ not in ['replicates', 'Concentration', 'Other'] and len(groups.loc[id_].unique()) > 1]
        if len(unique):
            # Group by the attributes specified above.
            grp = groups.fillna("-").T.groupby(unique).groups
        else:
            # Everything is in one group
            grp = {0: range(len(groups.T))}
        grp_formatted = {i: groups.iloc[:, id_] for i, (g, id_) in enumerate(grp.items())}
        return grp_formatted

    def get_rsd(self):
        # swap 0s and nan to avoid considering the mask
        main_grp = self.df.replace(0, np.nan).groupby('dt')
        df_rsd = 100 * main_grp.std() / main_grp.mean()
        if self.sub_zero:
            df_temp = df_rsd.fillna(method='bfill')  # Avoid subtracting NaN
            df_rsd_zero = df_rsd - df_temp.iloc[0]
            return df_rsd_zero
        else:
            return df_rsd

    def map_to_group(self, group):
        vals = group.loc[['replicates', 'Concentration']].values
        maps = {}
        concs = []
        for i, j in zip(vals[0, ...], vals[1, ...]):
            concs.append(f'{j}: ' + ','.join(i))
            for x, y in product(i, [j]):
                # The keys in maps are the well-coordinates (e.g, B1), and the values are the concentratios
                maps[x] = y

        mapped_df = self.df.loc[:, maps].copy()
        mapped_df.columns = [maps[col] for col in mapped_df.columns]
        mapped_df.columns.names = ['Concentration']
        # mapped_df = mapped_df.applymap(lambda well: well.feature_vec[0] if isinstance(well, Well) else -1) # TODO: Apply this elsewhere

        meta_table = group.iloc[1:, 0].to_frame()
        meta_table.iloc[0] = '\n'.join(concs)

        stacked = self.df.loc[:, maps].stack()
        stacked.index.names = stacked.index.names[:-1] + ['Well']
        return maps, mapped_df, meta_table, stacked


    def map_all(self):
        mapped_groups = {}
        for group_key, group in self.sub_groups.items():
            maps, mapped_df, meta_table, stacked = self.map_to_group(group)
            mapped_groups[group_key] = {'df': mapped_df,
                                 'maps': maps,
                                 'meta_table': meta_table,
                                 'well_images': stacked}
        return mapped_groups
