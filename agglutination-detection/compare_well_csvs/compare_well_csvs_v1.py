import pandas as pd
import argparse
import matplotlib.pyplot as plt
import ast
import sys
sys.path.insert(0,'..')
from plotting_and_visuals.plot_well_matrix import get_stats
import numpy as np
import scipy.stats as stats
from matplotlib import gridspec

# Visual variables for plotting
CONC_COLORS = ['#3A76AF','#EF8536','#519D3E','#C53932', '#8D6BB8', '#84584E']

# Variables for doing the t-test
TTEST_ALPHA = 0.05
NUM_STDEV_FOR_CI = 2

def main(args):
    well_data = pd.read_csv(args.experiment_dir+'/data.csv')

    # Metrics which we will plot for
    metrics = ['rsd', 'ml_agg_score', 'total_area_of_all_blobs']

    # In the .csv, the number of concentration levels will be denoted by rsd.1, rsd.2, ..., rsd.n, etc
    CONC_LVL_INDS = [int(col.split('.')[-1]) for col in list(well_data.columns) if "rsd." in col]

    # This helps us get the number of groups in the .csv
    sample_metric_str = f'rsd.{CONC_LVL_INDS[-1]}'
    GROUP_INDS = [i for i in range(2, len(well_data[sample_metric_str]))]

    # Time interval of consideration during the ttest
    time_interval = 5
    for metric in metrics:
        for group in GROUP_INDS:
            metrics_dict = {}
            
            # Plotting variables
            fig = plt.figure(figsize=(7, 7 + 4*len(CONC_LVL_INDS))) 
            gs = gridspec.GridSpec(3 + len(CONC_LVL_INDS), 1)
            ax_aggscore = plt.subplot(gs[0:3, :])
            axes_lod = [plt.subplot(gs[i, :]) for i in range(3, 3 + len(CONC_LVL_INDS))]

            for conc_lvl in CONC_LVL_INDS:
                metrics_dict[conc_lvl] = _plot_well_line(well_data, metric, group, conc_lvl, ax_aggscore)

            leg = plt.legend()
            ax_aggscore.set_title(args.experiment_dir.split("/")[-2])
            ax_aggscore.set_ylabel(f"Agglutination Score {metric}, a.u.")
            ax_aggscore.set_xlabel(f"Time (min)")

            control_lst = metrics_dict[CONC_LVL_INDS[0]]

            # Record an positive count plot for the given CI at a certain time interval
            num_timesteps = len(control_lst)
            pos_counts = np.zeros(num_timesteps)
            pos_counts_num_wells = np.zeros(num_timesteps)

            # Perform a two-sample T-test at every conc_lvl against the control
            reverse_conc_lvl_inds = list(CONC_LVL_INDS)
            reverse_conc_lvl_inds.reverse()
            for i, nonzero_conc_lvl in enumerate(reverse_conc_lvl_inds[:-1]):
                ax_lod = axes_lod[i]
                conc_lst = metrics_dict[nonzero_conc_lvl]
                significant_t = -1
                significant_time_interval = 0
                control_ci = get_ci(control_lst)
                for t in range(num_timesteps):
                    control_metrics_at_t = control_lst[t]
                    conc_metrics_at_t = conc_lst[t]
                    ttest_results = stats.ttest_ind(a=control_metrics_at_t, b= conc_metrics_at_t, equal_var=False)
                    pval = ttest_results.pvalue

                    # In order to consider a point in time to be significantly different, we need to see that difference for the last N minutes.
                    if significant_t == -1:
                        if pval < TTEST_ALPHA:
                            significant_time_interval += 1
                        else:
                            significant_time_interval = 0
                        if significant_time_interval == time_interval:
                            significant_t = t

                    # Count the number of wells that are outside the CI
                    control_ci_at_t = control_ci[t]
                    pos_counts_num_wells[t] += len(conc_metrics_at_t)
                    for conc_metric in conc_metrics_at_t:
                        if conc_metric > control_ci_at_t[1]:
                            pos_counts[t] += 1

                conc = leg.texts[nonzero_conc_lvl - 1].get_text()
                if significant_t != -1:
                    _plot_t_line(significant_t, color=CONC_COLORS[nonzero_conc_lvl - 1])
                    leg.get_texts()[nonzero_conc_lvl - 1].set_text(f"{conc} *t={significant_t}")

                ax_lod.subplot(1 + len(CONC_LVL_INDS), 1, 1 + (i + 1))
                pos_counts_acc = pos_counts / pos_counts_num_wells
                ax_lod.plot(pos_counts_acc)
                ax_lod.ylabel(f"True positive rate (LOD={conc})")
                ax_lod.xlabel(f"Time (min)")

            ax_aggscore.tight_layout()
            print(args.experiment_dir + f'/ttest_group_{group-2}_analysis_{metric}.svg')
            ax_aggscore.savefig(args.experiment_dir + f'/ttest_group_{group-2}_analysis_{metric}.svg')
            ax_aggscore.clf()
            break

def _plot_t_line(significant_t, color):
    plt.axvline(x=significant_t, color=color, linewidth=4)


def get_ci(metric_lst):
    metric_means = np.mean(metric_lst, axis=1)
    metric_stdevs = np.std(metric_lst, axis=1)
    metric_ci_lower = metric_means - NUM_STDEV_FOR_CI*metric_stdevs
    metric_ci_upper = metric_means + NUM_STDEV_FOR_CI*metric_stdevs
    metric_ci = np.array([[metric_ci_lower[i], metric_ci_upper[i]] for i in range(len(metric_ci_lower))])
    return metric_ci


def _plot_well_line(well_data, metric, group, conc_lvl, ax_aggscore):
    metric_str = f"{metric}.{conc_lvl}"
    concentration = well_data[metric_str][0]

    metric_lst = np.array(ast.literal_eval(well_data[metric_str][group]))
    metric_means = np.mean(metric_lst, axis=1)
    metric_stdevs = np.std(metric_lst, axis=1)
    ax_aggscore.errorbar(x=range(len(metric_means)), y=metric_means, yerr=metric_stdevs, label=concentration, marker='o', linestyle='', capsize=2, color=CONC_COLORS[conc_lvl-1])
    if concentration == '0.0':
        control_ci = get_ci(metric_lst)
        ax_aggscore.fill_between(range(len(metric_means)), control_ci[:, 1],  control_ci[:, 0], color='gray', alpha=0.95)
    return metric_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="For running different kinds of feature-extraction experiments on LAT trays.")
    parser.add_argument('--experiment_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
