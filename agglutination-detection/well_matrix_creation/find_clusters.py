import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from collections import Counter
from well_matrix import Well

def find_clusters(well_trays):
    """
        TODO: Fill out docs and re-comment
        Group images of wells based on their location in the image
    """
    # A list of coordinates for each frame
    well_locs = [tray.mapper for tray in well_trays]
    # Stores all coordinates in all frames
    X = np.concatenate(well_locs)
    # Count the number of coordinates in each frame
    well_count = np.array([len(loc) for loc in well_locs])
    # (x, y) is actually the center of the circle in the tray domain space.
    # Here, we group wells across frames with similar (x, y) in the same cluster.
    # If we're missing some wells due to hough circles failing, we'll just get
    # less members in that cluster.
    if well_count.max() < 96:
        cluster = KMeans(n_clusters=96)
    else:
        cluster = KMeans(n_clusters=well_count.max())

    # Create a set of reference coordinates
    med_x = np.median(np.diff(X[..., 0]))
    xs = X[..., 0]
    xs = np.median(xs[xs < np.quantile(xs, 1 / 12)])
    ys = X[..., 1]
    ys = np.median(ys[ys < np.quantile(ys, 1 / 8)])

    x_ = np.arange(xs, xs + med_x * 12, med_x)
    y_ = np.arange(ys, ys + med_x * 8, med_x)
    xx, yy = np.meshgrid(x_, y_, sparse=False, indexing='xy')
    xy = np.stack((xx, yy), axis=-1)
    xy = xy.reshape(96, 2).astype('int32')

    # Maps each well-coordinate to a cluster.
    # NOTE: X is a list of coordinates across all frames, and y_preds match each coordinate inside to a cluster.

    cluster.fit(np.concatenate((xy, X)))
    y_pred = cluster.predict(X)

    # Count instances of each cluster
    y_cnt = Counter(y_pred)

    # Group clusters
    over = []
    under = []
    exact = []
    thresh = len(well_trays)
    for k in y_cnt:
        if y_cnt[k] > thresh:
            over.append(k)
        elif y_cnt[k] < 0.6 * thresh:
            under.append(k)
        else:
            exact.append(k)

    # Locate wells that are slightly shifted and correct their position
    if len(over) or len(under):
        k = exact[0]
        diff = X[y_pred == k] - np.median(X[y_pred == k], axis=0)
        for j in np.where((abs(diff).sum(axis=1) > 60))[0]:
            well_locs[j] = well_locs[j] - diff[j]
        # xy = xy - diff[j]

        X = np.concatenate(well_locs)
        if well_count.max() < 96:
            cluster = KMeans(n_clusters=96)
        else:
            cluster = KMeans(n_clusters=well_count.max())

        # Fit the model again
        cluster.fit(np.concatenate((xy, X)))
        # y_pred = cluster.predict(X)

    ''' Deprecated
    # Get the number of wells before the max frame
    idx_max = well_count.argmax()
    if idx_max > 0:
        idx_max = well_count.cumsum()[idx_max - 1]

    # y_pred_max denotes the predictions on the maximum frame.
    y_pred_max = y_pred[idx_max:idx_max + well_count.max()]
    # new_map maps each index in y_pred_max to a cluster_id. there are is an assumption here:
    #  every element in y_pred_max is unique, and in fact, y_pred_max is a permutation of range(len(y_pred_max)).
    contains_all_unique = len(np.unique(y_pred_max)) == len(y_pred_max)
    possible_cluster_ids = range(len(y_pred_max))
    is_permutation_of_range = Counter(possible_cluster_ids) == Counter(y_pred_max)
    if not contains_all_unique or not is_permutation_of_range:
        raise Exception(f"Error with finding clusters.")
    # now keep in mind, the indices of y are just a permutation of the possible_cluster_ids.
    # here, we're mapping each cluster_id to a y index.
    # new_map = {cluster_id: index_in_y + 1 for index_in_y, cluster_id in enumerate(y_pred[idx_max:idx_max + well_count.max()])}
    '''

    # create a map based on the reference coordinates
    new_map = {ys: j + 1 for j, ys in enumerate(cluster.predict(xy))}
    # now, we match each coordinate's assigned cluster to an index in y_pred_max.
    # all_columns = [new_map.get(i, f'U{i}') for i in y_pred]

    dfs = {}
    for well_tray in well_trays:
        frame = well_tray.to_frame()
        columns = np.array([list(col) for col in frame.columns])
        columns = [new_map.get(i, f'U{i}') for i in cluster.predict(columns)]
        frame.columns = columns
        frame = frame.loc[:, ~frame.columns.duplicated(keep='last')]
        # Transforms this frame into a dataframe
        # Right side: maps the tray into a dataframe, where the columns are the assigned_map
        # Left side: Has keys of image and timestamp
        # Assigning two indices, and the frame has a third (pixel)
        dfs[(well_tray.path.split(os.sep)[-1], well_tray.date_obj)] = frame
    # Creates one big dataframes from all frames
    dfs = pd.concat(dfs)
    # The 3D nature of the dataframe
    dfs.index.names = ['Image', 'Timestamp', 'Frame']
    # Sort entries by timestamp
    dfs = dfs.sort_index(level='Timestamp').reset_index()
    # Now that the entries are sorted by timestamp, re-assign the frame index to match accordingly.
    num_frames = len(dfs['Timestamp'].index)
    dfs['Frame'] = range(num_frames)
    for frame in range(num_frames):
        wells_in_frame = dfs.query(f'Frame == {frame}').values[0]
        for well in wells_in_frame:
            if not isinstance(well, Well):
                continue
            well.frame = frame

    # Assigning a fourth index type
    dfs['dt'] = (dfs['Timestamp'] - dfs.loc[0, 'Timestamp']) / np.timedelta64(1, 'm')
    dfs = dfs.set_index(['Image', 'Timestamp', 'dt', 'Frame'])
    return dfs



