import numpy as np
import joblib
from sklearn.preprocessing import normalize
from well_matrix import WellMatrix, Well
import sys
sys.path.append('training_ml')
from ngboost import NGBRegressor
# from gamma import Gamma
import training_ml.lognormal
from training_ml.lognormal import LogNormal
import numpy as np
import scipy as sp
from scipy.stats import lognorm as dist
from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore
import pickle
from pathlib import Path


def load_ml_model(ml_model_type, model_dir='.'):
    print("Loading ML model")
    if ml_model_type == 'ngboost':
        print("Using NGBoost for predictions")
        file_path = Path(f"{model_dir}/ml_{ml_model_type}.joblib.pkl")
        with file_path.open("rb") as f:
            ml_model = pickle.load(f)
    else:
        ml_model = joblib.load(f'{model_dir}/ml_{ml_model_type}.joblib.pkl')
    return ml_model


def ml_predict(well_matrix: WellMatrix, ml_model_type: str, mean_x: float, var_x: float) -> None:
    print("Loading ML model")
    ml_model = load_ml_model(ml_model_type)

    print("Predicting agglutination score")
    X, y = [], []
    # Go through each well, and assign it an agglutination score
    for frame in range(well_matrix.shape[2]):
        for well_coord in well_matrix.all_selected_well_coords:
            curr_well = well_matrix[well_coord, frame]
            if not isinstance(curr_well, Well):
                continue
            well_feature_vec = curr_well.feature_vec
            X.append(well_feature_vec)

    # Impute the NaN values in X with their normalized nanmean values.
    X = np.array(X)
    for col in range(len(X[0])):
        nan_inds = np.isnan(X[:, col])
        X[nan_inds, col] = mean_x[col]

    # Normalize the data
    X -= mean_x
    X /= var_x

    # Now, we will likely want to plot the agglutination predictions. To do so, we 
    # include the predictions as the last entry in each Well's feature vector.
    y = ml_model.predict(X)
    if ml_model_type == 'ngboost':
        y_dists = ml_model.pred_dist(X)

    i = 0
    for frame in range(well_matrix.shape[2]):
        for well_coord in well_matrix.all_selected_well_coords:
            curr_well = well_matrix[well_coord, frame]
            if not isinstance(curr_well, Well):
                continue
            curr_well_agg_score_prediction = y[i]
            if ml_model_type == 'ngboost':
                curr_well_agg_score_prediction -= 1
            curr_well_agg_score_prediction = np.clip(curr_well_agg_score_prediction, 0, 5)
            new_well_feature_vec = curr_well.feature_vec.tolist()
            new_well_feature_vec.append(curr_well_agg_score_prediction)
            curr_well.feature_vec = np.array(new_well_feature_vec)
            if ml_model_type == 'ngboost':
                curr_well.agg_score_dist = y_dists[i]
            i += 1


def add_ground_truth(well_matrix: WellMatrix):
    for frame in range(well_matrix.shape[2]):
        for well_coord in well_matrix.all_selected_well_coords:
            curr_well = well_matrix[well_coord, frame]
            if not isinstance(curr_well, Well):
                continue
            new_well_feature_vec = curr_well.feature_vec.tolist()
            new_well_feature_vec.append(curr_well.label)
            curr_well.feature_vec = np.array(new_well_feature_vec)