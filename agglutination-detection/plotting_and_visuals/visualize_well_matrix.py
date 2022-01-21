import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
from tqdm import tqdm
from well_matrix import Well



def plot_well_matrix_over_time(well_matrix, experiments_dir, image_folder_name, visual_name, rgb=False):
    """
        Plots the selected wells where each frame is represented in a row.
    """
    if rgb:
        print("Plotting images as RGB")
    # Variables for processing
    well_img_radius, num_frames, groups_data = well_matrix.well_img_radius, well_matrix.shape[2], well_matrix.main_df_subgrouped
    max_wells_in_a_group = max([len(groups_data[group]['df'].columns) for group in groups_data])
    # Create a white image which we will use to track the time
    white_label_img = np.ones((well_matrix.well_img_radius * 2, well_matrix.well_img_radius * 2* max_wells_in_a_group))
    plt.figure(figsize=(int(8.5*max_wells_in_a_group), int(1.2*num_frames)))
    # Used as a counting variable for subplots
    well_plotted = 1
    # Iterate through all rows in the plot 
    for frame in tqdm(range(num_frames)):
        # Iterate through all columns (groups) in this row
        for group in groups_data:
            group_data = groups_data[group]
            # For each group, create an image that groupatenates all the wells at this frame
            group_img = np.ones((well_matrix.well_img_radius*2, well_matrix.well_img_radius*2 * max_wells_in_a_group))
            if rgb:
                group_img = np.repeat(group_img[:, :, np.newaxis], 3, axis=2)
            well_coords_in_group = group_data['maps'].keys()
            for i, well_coord in enumerate(well_coords_in_group):
                curr_well = well_matrix[well_coord, frame]
                if not isinstance(curr_well, Well):
                    continue
                if rgb:
                    group_img[:, i * well_img_radius*2 : (i+1) * well_img_radius*2, :] = curr_well.get_image_alias()
                else:
                    group_img[:, i * well_img_radius*2 : (i+1) * well_img_radius*2] = curr_well.get_image_alias()

            # Plot the constructed image
            plt.subplot(num_frames, len(groups_data) + 1, well_plotted)
            if frame == 0:
                plt.title(f"group {group}_{'_'.join(well_coords_in_group)}")
            plt.axis('off')
            if rgb:
                plt.imshow(group_img)
            else:
                plt.imshow(group_img, cmap='gray')
            well_plotted += 1
        # Add a frame label
        curr_dt = well_matrix.main_df_subgrouped[0]['df'].index.get_level_values('dt')[frame]
        frame_label_img = cv2.putText(white_label_img.copy(), "t={:.2f}min".format(curr_dt), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                        5, (0, 0, 0), 5, cv2.LINE_AA)
        plt.subplot(num_frames, len(groups_data) + 1, well_plotted)
        plt.axis('off')
        plt.imshow(frame_label_img, cmap='gray')
        well_plotted += 1
    plt.tight_layout()
    plt.savefig(f"{experiments_dir}/well_matrix_{image_folder_name}_{visual_name}")



# def plot_wells(self, group, show_plots=None, save_plots=None):
#     stacked_wells = self.stats[group]['well_images'].groupby(['dt', 'Well'])
#     cols = np.array(list(self.stats[group]['maps'].keys()))
#     rows = self.stats[group]['df'].index.values
#     nrows = len(rows)
#     ncols = len(cols)

#     fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
#                             figsize=(ncols, nrows))

#     [(ax.cla(), ax.set_xticks([]), ax.set_yticks([])) for ax in axs.ravel()]

#     for well, dw in stacked_wells:
#         # print(well)
#         # print("=============")
#         # print(dw)
#         x, y = (np.where(rows == well[0])[0][0], np.where(cols == well[1])[0][0])
#         ax = axs[x, y]
#         if x == 0:
#             ax.set_title(cols[y], fontsize=nrows / 2.5)
#         if y == 0:
#             ax.set_ylabel(f'{well[0]:0.0f}', rotation='horizontal',
#                             fontsize=nrows / 2.5, ha='right', va="center")
#         img = dw.values
#         print(img)
#         img = img.well_image
#         dim = int(img.shape[0] ** 0.5)
#         ax.imshow(img.reshape(dim, dim), 'gray', aspect='auto')

#     if show_plots is None:
#         show_plots = self.show_plots
#     if save_plots is None:
#         save_plots = self.save_plots

#     if show_plots:
#         fig.show()
#     if save_plots:
#         fig.savefig(self.folder + f'Wells_group{group}.svg')

