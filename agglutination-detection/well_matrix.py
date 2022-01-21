import numpy as np
import pandas as pd
from helpers import POSITION_MAP

class Well:
    """
        Defines an Well instance, which represents a single well
        in the agglutination tray. 
        Attributes:
            - self.well_image: The image that represents this well. 
            - self.feature_vec: The feature vector that represents this well. This is used to compute it's agglutination score.
            - self.well_saved_states: A list containing copies of the `self.well_image`, as it goes through it's image processing steps. 
                This is optionally stored if it's needed to compute a feature; otherwise it's set to None.
            - self.contours: A list of contours of the well. Set later in preprocessing.
            - self.agg_score_dist: (Only set with NGBoost) the learned distribution of the agglutination scores
    """
    def __init__(self, well_image, feature_vec_size, type):
        """
            Instantiates a Well object.
        """
        self.well_image = well_image
        self.feature_vec = np.zeros(feature_vec_size)
        self.contours = None
        self.well_saved_states = {}
        self.label = -1
        self.type = type
        self.agg_score_dist = None


    def get_image_alias(self):
        """
            Return an alias of this Well's image representation. Note that mutating this alias
            will mutate this attribute.
        """
        return self.well_image


    def update_feature_vec(self, feature_val, feature_ind):
        """
            Update this Well's feature vector.
        """
        self.feature_vec[feature_ind] = feature_val


    def set_contours(self, contours):
        """
            Update the contours in this well.
        """
        self.contours = contours


class WellMatrix():
    """
        Defines an WellMatrix instance, which represents an agglutination tray
        and it's selected wells for processing.

        Attributes:
            - self.well_img_radius: The radius of each well image. Each well image will be a square with sidelength self.well_img_radius.
            - self.all_selected_well_coords: A (N, 2) matrix that contains the (y, x) coordinates of all selected
                wells for processing.
            - self.shape: The shape of the WellMatrix representation. It is of shape (WELL_TRAY_WIDTH, WELL_TRAY_HEIGHT, NUM_FRAMES)
            - self.main_df: The actual data-structure implemented for this WellMatrix. We use a 'matrix-like' interface,
                however, the actual implementation is a dataframe.
            - self.main_df_subgrouped: The same `self.main_df` as above, except it groups Wells together that belong to the same
                group (as specified in the meta.yml file)
    """

    def __init__(self, main_df: pd.DataFrame, main_df_subgrouped: pd.DataFrame) -> None:
        """
            Arguments:
                - main_df: A dataframe of Well objects without any subgrouping
                - main_df: A dataframe of Well objects organized in different groups
        """
        self.main_df = main_df
        self.main_df_subgrouped = main_df_subgrouped
        self.shape = (8, 12, len(self.main_df.index.get_level_values('Frame')))
        self.all_selected_well_coords = list(main_df.columns)
        self.groups = self.main_df_subgrouped.keys()
        self.well_img_radius = 91


    def __getitem__(self, indices: str or tuple) -> Well:
        """
            Allows us to do matrix indexing with a WellMatrix object.
        """
        if type(indices) == str:
            indices = (indices)
        dim = len(indices)
        well_coord, frame = indices[0], int(indices[1])
        well_over_time = self.main_df[well_coord].values
        if dim == 1:
            return well_over_time
        elif dim == 2:
            return well_over_time[frame]
        else:
            raise IndexError("The WellMatrix is 2D. The first dimension is the coordinate (e.g, A1) and the second dimension is the frame.")
