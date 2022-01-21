import matplotlib.pyplot as plt
from well_matrix import WellMatrix, Well
import numpy as np
import os
from helpers import compute_fixed_attrs_str
from scipy.signal import savgol_filter
import pandas as pd


def plot_smoothed_scores(well_matrix: WellMatrix, image_folder_name: str, experiment_dir: str, GET_FEATURE_IND: dict):
    for group in well_matrix.groups:
        gt_ind = GET_FEATURE_IND["ground_truth"]
        pred_ind = GET_FEATURE_IND["ml_agg_score"]
        mapped_df_gt = well_matrix.main_df_subgrouped[group]['df'].applymap(lambda well: well.feature_vec[gt_ind] if isinstance(well, Well) else -1)
        mapped_df_preds = well_matrix.main_df_subgrouped[group]['df'].applymap(lambda well: well.feature_vec[pred_ind] if isinstance(well, Well) else -1)

        concs = list(np.unique(mapped_df_preds.columns))
        for conc in concs:
            gts = mapped_df_gt[conc].values.T
            preds = mapped_df_preds[conc].values.T
            for i in range(len(gts)):
                well_gt, well_pred = gts[i], preds[i]
                pred_xs = range(1, len(well_pred) + 1)
                gt_xs = [i+1 for i in range(len(well_gt)) if well_gt[i] != -1]
                well_gt = [gt for gt in well_gt if gt != -1]
                plt.clf()
                plt.figure(figsize=(10,10))
                plt.subplot(2, 1, 1)
                plt.plot(gt_xs, well_gt, label='Ground truth')
                plt.plot(pred_xs, well_pred, label='Prediction')
                plt.scatter(gt_xs, well_gt, label='Ground truth')
                plt.scatter(pred_xs, well_pred, label='Prediction')
                plt.ylim(0, 5)
                plt.legend()
                plt.rcParams.update({'font.size': 10})
                plt.title(f"In {image_folder_name}, for group {group}, well {i}, with concentration {conc}")

                plt.subplot(2, 1, 2)
                window_len = 9
                type = "SMOOTHED"
                if len(well_gt) < window_len:
                    well_gt_smooth = well_gt
                    type = "PREDICTIONS ONLY SMOOTHED"
                else:
                    well_gt_smooth = savgol_filter(well_gt, window_len, 3) 
                well_pred_smooth = savgol_filter(well_pred, window_len, 3)
                plt.plot(gt_xs, well_gt_smooth, label='Ground truth')
                plt.plot(pred_xs, well_pred_smooth, label='Prediction')
                plt.scatter(gt_xs, well_gt_smooth, label='Ground truth')
                plt.scatter(pred_xs, well_pred_smooth, label='Prediction')
                plt.ylim(0, 5)
                plt.legend()
                plt.rcParams.update({'font.size': 10})
                plt.title(f"In {image_folder_name}, for group {group}, well {i}, with concentration {conc}, {type}")
                plt.savefig(experiment_dir + f'/{image_folder_name}_group_{group}_well_{i}_conc_{conc}.png')


def ngboost_interval_plot(well_matrix: WellMatrix, image_folder_name: str, experiment_dir: str, GET_FEATURE_IND: dict):
    for group in well_matrix.groups:
        pred_ind = GET_FEATURE_IND["ml_agg_score"]
        mapped_df_preds = well_matrix.main_df_subgrouped[group]['df'].applymap(lambda well: well.feature_vec[pred_ind] if isinstance(well, Well) else -1)
        mapped_df_dists = well_matrix.main_df_subgrouped[group]['df'].applymap(lambda well: well.agg_score_dist if isinstance(well, Well) else -1)

        concs = list(np.unique(mapped_df_dists.columns))
        for conc in concs:
            preds = mapped_df_preds[conc].values.T
            dists = mapped_df_dists[conc].values.T
            for i in range(len(preds)):
                well_pred = [pred for pred in preds[i] if pred != -1]
                well_dists = [dist for dist in dists[i] if dist != -1]
                pred_xs = range(1, len(well_pred) + 1)
                plt.clf()
                plt.figure(figsize=(10,10))
                plt.plot(pred_xs, well_pred, label='Prediction')
                plt.scatter(pred_xs, well_pred, label='Prediction')
                plt.ylim(0, 5)
                plt.legend()
                plt.rcParams.update({'font.size': 10})
                plt.title(f"In {image_folder_name}, for group {group}, well {i}, with concentration {conc}")

                predictions_upper = [well_dist.dist.interval(0.95)[1] for well_dist in well_dists]
                predictions_lower = [well_dist.dist.interval(0.95)[0] for well_dist in well_dists]
                # predictions_upper = pd.DataFrame(well_dist.dist.interval(0.95)[1], columns=['Predictions_upper'])
                # predictions_lower = pd.DataFrame(well_dist.dist.interval(0.95)[0], columns=['Predictions_lower'])
                plt.fill_between(pred_xs, predictions_lower,  predictions_upper,label = '95% Prediction Interval', color='gray', alpha=0.5)
                plt.savefig(experiment_dir + f'/{image_folder_name}_group_{group}_well_{i}_conc_{conc}.png')


def get_stats(df, by='Concentration'):
    m_grp = df.T.groupby(by)
    mean = m_grp.mean().T
    std = m_grp.std().T
    count = m_grp.count().max(axis=1)
    count.index = [str(col) for col in count.index]
    return mean, std, count


def plot_table(main_df, group, ax):
    bbox = [0, 0, 1, 1]
    meta_table = main_df[group]['meta_table']
    ax.set_axis_off()
    mpl_table = ax.table(cellText=meta_table.values, rowLabels=meta_table.index,
                         bbox=bbox, loc='center')
    mpl_table.auto_set_font_size(False)
    table_font_size = 9  # 11
    mpl_table.set_fontsize(table_font_size)
    return mpl_table


def plot_scores(main_df, mean, std, count, group, folder, plot_name,
                fig, ax, mpl_table, dt_limit=1000):
    # if folder[-1] != '/':
    #     folder += '/'
    meta_table = main_df[group]['meta_table']

    try:
        mean_ = mean.reset_index(level='dt').reset_index(drop=True)
        mean_ = mean_.loc[mean_['dt'] <= dt_limit].set_index('dt')
        std_ = std.reset_index(level='dt').reset_index(drop=True)
        std_ = std_.loc[std_['dt'] <= dt_limit].set_index('dt')
    except ValueError:
        mean_ = mean
        std_ = std
    mean_.plot(ax=ax, marker='o', linestyle='', yerr=std_, capsize=2)

    # ax_x_ticks = ["{:.0f}".format(dt) for dt in mean.index.get_level_values('dt')]
    # ax.set_xticks(range(len(ax_x_ticks)))
    # ax.set_xticklabels(ax_x_ticks)

    ax.set_title(folder.split('/')[-2])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(f'Agglutination Score ({plot_name}), a.u.')
    handles, labels = ax.get_legend_handles_labels()
    labels = [l + f' (n={count.loc[l]})' for l in labels]
    ax.legend(labels=labels, handles=handles, title="SARS-CoV-2, PFU/mL")
    # ax[1].set_axis_off()
    # bbox = [0, 0, 1, 1]
    # mpl_table = ax[1].table(cellText=meta_table.values, rowLabels=meta_table.index,
    #                         bbox=bbox, loc='center')
    # mpl_table.auto_set_font_size(False)
    # table_font_size = 5.5
    # mpl_table.set_fontsize(table_font_size)

    concs = meta_table.iloc[0, 0].split('\n')
    if len(concs) > 1:
        cell_dict = mpl_table.get_celld()
        h = cell_dict[(0, 0)].get_height()
        [cell_dict[(0, i)].set_height(h * (len(concs) - 1)) for i in range(-1, 1)]

    fig.tight_layout()


def wells_for_csv(df, meta, dir):
    meta_idx = meta.iloc[1:-1].T.copy()
    meta_idx['Experiment'] = dir.split('/')[-2]
    meta_idx = meta_idx.set_index('Experiment')
    df = df.reset_index('dt')
    for col in df.columns:
        meta_idx[col] = [df[col].values.tolist()]
    meta_idx = meta_idx.set_index([col for col in meta_idx.columns if col not in df.columns], append=True)
    return meta_idx

    # if show_plots:
    #     fig.show()

    # if save_plots:
    #     fig.savefig(folder + f'group{group}_{plot_name}.svg')

def plot_wells(main_df, group, folder, plot_wells_at_times, show_plots=None, save_plots=None):
    stacked_wells = main_df[group]['well_images'].groupby(['dt', 'Well'])
    cols = np.array(list(main_df[group]['maps'].keys()))
    rows = main_df[group]['df'].index.values
    rows = np.array([list(row) for row in rows])
    rows = rows[:, 2]
    # Keep only the rows that are in <plot_wells_at_times>
    rows = [dt for dt in rows if int(dt) in plot_wells_at_times]

    nrows = len(rows)
    ncols = len(cols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(ncols , nrows))

    [(ax.cla(), ax.set_xticks([]), ax.set_yticks([])) for ax in axs.ravel()]

    for well, dw in stacked_wells:
        well_rows = np.where(rows == well[0])[0]
        if len(well_rows) == 0:
            # This means we should not plotting at this timepoint
            continue

        well_cols = np.where(cols == well[1])[0]
        well_row_coord = well_rows[0]
        well_col_coord = well_cols[0]
        
        x, y = (well_row_coord, well_col_coord)
        ax = axs[x, y]
        if x == 0:
            ax.set_title(cols[y], fontsize=14)
        if y == 0:
            ax.set_ylabel(f'{well[0]:0.0f}', rotation='horizontal',
                            fontsize=14, ha='right', va="center")
        well = dw.values[0]
        well_img = well.get_image_alias()
        dim = int(well_img.shape[0])
        ax.imshow(well_img.reshape(dim, dim), 'gray', aspect='auto')

    if show_plots:
        fig.show()
    if save_plots:
        fig.savefig(folder + f'Wells_group{group}.svg')

# def plot_all(group):
#     plot_scores(group)
#     plot_wells(group)
