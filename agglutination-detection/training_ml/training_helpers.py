from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import sys
sys.path.insert(0,'..')
from constants import NUM_CLASSES
from helpers import get_config
from experiment_runner import ExperimentRunner
from ngboost import NGBRegressor
# from gamma import Gamma
from lognormal import LogNormal

EPS = 0.001


def get_all_models_dict(ml_params):
    if ml_params['model_type'] == 'classifier':
        model_params = ml_params['model_params']
        return {
            'SGDClassifier': lambda: SGDClassifier(),
            'GaussianNB': lambda: GaussianNB(),
            'RandomForestClassifier': lambda: RandomForestClassifier(n_estimators=model_params['rf']['n_estimators'], max_depth=model_params['rf']['max_depth']),
            'MLPClassifier': lambda: MLPClassifier(hidden_layer_sizes=tuple(model_params['mlp']['hidden_layer_sizes']), alpha=model_params['mlp']['alpha'], max_iter=model_params['mlp']['max_iter']),
            'AdaBoostClassifier': lambda: AdaBoostClassifier(),
            'GradientBoostingClassifier': lambda: GradientBoostingClassifier()
        }
    else:
        model_params = ml_params['model_params']
        return {
            # 'SGDRegressor': lambda warmstart: SGDRegressor(),
            # 'RandomForestRegressor': lambda warmstart: RandomForestRegressor(max_depth=6, n_estimators=260, warm_start=warmstart),
            # 'MLPRegressor': lambda warmstart: MLPRegressor(alpha=model_params['mlp']['alpha'], max_iter=model_params['mlp']['max_iter'], warm_start=warmstart),
            # 'AdaBoostRegressor': lambda warmstart: AdaBoostRegressor(n_estimators=10),
            # 'GradientBoostingRegressor': lambda warmstart: GradientBoostingRegressor(max_depth=4, n_estimators=10 if warmstart else 135, warm_start=warmstart),
            # 'NGBRegressor': lambda warmstart: NGBRegressor(Dist=Gamma),
            'NGBRegressor': lambda warmstart: NGBRegressor(Dist=LogNormal),
            # 'LinearRegression': lambda warmstart: LinearRegression(),
            # 'CNN': lambda: None, # CNN models get instantiated during training for a gridsearch
            # 'custom_mlp_model': custom_mlp_model()
        }
    # else:
    #     raise Exception("Invalid model_type in .yml file")


def acc_prec_recall(C):
    ''' Compute accuracy, precision, and recall given Numpy array confusion matrix C. Returns a floating point value '''
    diag_sum = C.diagonal().sum()
    overall_sum = C.sum()
    acc = diag_sum / overall_sum   
    prec = [C[i, i] / (np.sum(C[:, i] + EPS)) for i in range(NUM_CLASSES)]
    recall = [C[i, i] / np.sum(C[i, :] + EPS) for i in range(NUM_CLASSES)]
    return acc, prec, recall


def acc_recall_off_by_one(C):
    ''' Compute accuracy, precision, and recall given Numpy array confusion matrix C. Returns a floating point value 
        Same definitions as above, except off-by-one errors are okay.    
    '''
    diag_sum = C.diagonal().sum() + C.diagonal(1).sum() + C.diagonal(-1).sum()
    overall_sum = C.sum()
    acc = diag_sum / overall_sum   
    recall = [(C[0, 0] + C[0, 1]) /  np.sum(C[0, :])]
    recall.extend([(C[i, i] + C[i, i-1] + C[i, i+1]) / (np.sum(C[i, :] + EPS)) for i in range(1, NUM_CLASSES-1)])
    recall.append((C[4, 4] + C[4, 3]) /  (np.sum(C[4, :])+ EPS) )
    return acc, recall


def get_error_urls(true_labels, preds, X_test_urls, raw_preds):
    """
        Get the URLs where the model predicted incorrectly
    """
    preds = np.array(preds)
    raw_preds = np.array(raw_preds)
    true_labels = np.array(true_labels)
    X_test_urls = np.array(X_test_urls)
    mistake_str = ""
    for label in range(5):
        true_label_indices = np.where(true_labels == label)[0].astype(int)
        urls_for_label = X_test_urls[true_label_indices]
        pred_for_label = preds[true_label_indices]
        raw_preds_for_label = raw_preds[true_label_indices]
        mistake_indices = np.where(pred_for_label != label)[0].astype(int)
        mistake_urls = urls_for_label[mistake_indices]
        mistake_str += f"\n\n Mistakes for {label} \n"
        mistake_str += f"====================\n\n"
        for i in range(len(mistake_indices)):
            mistake_url, prediction, raw_pred = mistake_urls[i], pred_for_label[mistake_indices[i]], raw_preds_for_label[mistake_indices[i]]
            mistake_str += f"{mistake_url} | truth={label} prediction={prediction} raw_pred={raw_pred}\n"
            # mistake_str += f"{mistake_url},{label}\n"
            # mistake_str += f"{mistake_url}\n"
        mistake_str += f"\n====================\n"
    return mistake_str


def get_normalization_constants(X):
    """
        Returns the constants that normalize the dataset.
        Tested for correctness.
    """
    mean_x = np.mean(X, axis=0)
    var_x = np.var(X, axis=0)
    return mean_x, var_x


def run_experiment(labels, tray_name, args, opt):
    tray_folder_dirpath = f"{args.full_tray_imgs_dir}{os.sep}{tray_name}"
    config_filepath = f"{tray_folder_dirpath}{os.sep}meta.yml"
    well_metadata = get_config(None, config_filepath, load_only=True)
    runner = ExperimentRunner(opt, well_metadata, tray_folder_dirpath)
    runner._create_well_matrix(should_save_key_states=True, feature_vec_size=opt.feature_vec_size, saved_well_matrices_path='../_saved_well_matrices', matrix_filename=tray_name, labels=labels, resave=args.resave == 'true')
    runner.run(f"{tray_folder_dirpath}/experiments", saved_well_matrices_path='../_saved_well_matrices')
    return runner


