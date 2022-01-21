from operator import mod
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'..')
from constants import NUM_CLASSES, WARMSTART_MODELS
from helpers import get_config
# from cnn_model import cnn_model
# from custom_loss_mlp import custom_mlp_model
import shap as shap
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict
from training_helpers import get_all_models_dict, acc_prec_recall, acc_recall_off_by_one, get_error_urls
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score
import json
import uuid
import pickle
from pathlib import Path

EPS = 0.001

def compute_confusion_matrix(p1, p2):
    """Computes a confusion matrix using numpy for two np.arrays.
    Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"
    However, this function avoids the dependency on sklearn."""

    K = 5
    result = np.zeros((K, K))
    for i in range(len(p1)):
        p1_label, p2_label = int(p1[i]), int(p2[i])
        result[p1_label][p2_label] += 1
    return result


def get_kappas_score(p1, p2):
    p1, p2 = [int(np.round(p1_label)) for p1_label in p1], [int(np.round(p2_label)) for p2_label in p2]
    k_s = cohen_kappa_score(p1, p2)
    return k_s


class MLTrainer():

    def __init__(self, model_savepath, ml_config, dataset_config, warmstart, save_results, num_classes=5) -> None:
        self.model_savepath = model_savepath
        self.ml_params = ml_config.ml_params
        self.num_classes = num_classes
        self.dataset_config = dataset_config
        self.seed = ml_config.seed
        self.model_savename = self.ml_params['model_savename']
        self.model_type = self.ml_params['model_type']
        self.warmstart = warmstart
        self.save_results = save_results


    def _train(self, model, model_name, keras_model=False, warmstart=False):
        print_str = "Training model"
        if warmstart:
            print_str += " with warmstart"
        print(print_str)

        if keras_model:
            model.fit(self.imgs_train, self.y_train, steps_per_epoch=10000, epochs=4)
            preds = model.predict(self.imgs_test)
        else:
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            pred_dists = model.pred_dist(self.X_test)
        return preds


    def _eval(self):
        C = compute_confusion_matrix(self.y_test, self.preds)
        acc, prec, recall = acc_prec_recall(C)
        acc_off_by_one, recall_off_by_one = acc_recall_off_by_one(C)
        return acc, prec, recall, acc_off_by_one, recall_off_by_one, C


    def _write_model_report(self, model_report, hyperparams=None, write_mistakes_str=False):
        model_name, accuracy, precision, recall, acc_off_by_one, recall_off_by_one, conf_matrix, mistakes_str, model, cv_scores, kappas_coef = model_report
        with open(f"./ml_regressor_report.txt", "a") as outf:
            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {model_name}:\n')  
            outf.write(f'\tAccuracy: {accuracy:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
            outf.write(f'\tAccuracy discounting off-by-one errors: {acc_off_by_one:.4f}\n')
            outf.write(f'\tRecall discounting off-by-one errors: {[round(item, 4) for item in recall_off_by_one]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix.astype(int)}\n\n')
            outf.write(f'\tCV Scores: \n{[round(item, 4) for item in cv_scores]}\n\n')
            outf.write(f'\tCV Mean: \n{np.mean(cv_scores):.4f}\n\n')
            outf.write(f'\tKappas Coef Mean: \n{kappas_coef}\n\n')
            # outf.write(f'\t{model.summary()}\n\n')
            if model_name == 'CNN':
                model.summary(print_fn=lambda x: outf.write(x + '\n'))
                outf.write(f'\tHyperparams \n {hyperparams}\n\n')

            if model_name == 'GradientBoostingRegressor':
                outf.write(f'\timportances \n {model.feature_importances_}\n\n')
            
        if write_mistakes_str:
            with open(f"./mistake_urls/ml_regressor_report_{model_name}_mistakes.txt", "w") as outf:
                outf.write(f'\t{mistakes_str}\n\n')


    def begin_training(self) -> None:
        """
            Loads in the datasets, and runs the ML experiments.
        """
        # Precondition: The dataset is normalized
        dataset = np.load('./dataset.npz')
        # dataset_warm = np.load('./dataset_warmstart.npz')
        self.X, self.y, self.groups, self.imgs, self.X_urls = dataset['X'], dataset['y'], dataset['groups'], dataset['imgs'], dataset['X_urls']
        self.imgs = np.expand_dims(self.imgs, axis=-1) / 255
        # self.X_warm, self.y_warm, self.groups_warm, self.imgs_warm, self.X_urls_warm = dataset_warm['X'], dataset_warm['y'], dataset_warm['groups'], dataset_warm['imgs'], dataset_warm['X_urls']
        # self.imgs_warm = np.expand_dims(self.imgs_warm, axis=-1) / 255
        if self.ml_params['model_type'] == 'classifier':
            self.y = np.around(self.y)
        
        if self.model_type == 'ngboost':
            self.y += 1
        self.num_classes = len(np.unique(np.around(self.y)))
        self.run_ml_experiments(self.warmstart)


    def run_ml_experiments(self, warmstart=False):
        ''' This function performs the ML experiments. 
        '''
        all_models_cv = get_all_models_dict(self.ml_params)

        for model_name in all_models_cv:
            model_cv_fn = all_models_cv[model_name]
            print(f"Training {model_name} using cross validation")
            # Create the cross-validation split
            group_kfold = GroupKFold(n_splits=10)
            group_ksplit = group_kfold.split(self.X, self.y, self.groups)
            # We'll have a confusion matrix which gets accumulated, and we'll record accuracy at each fold.
            C = np.zeros((NUM_CLASSES, NUM_CLASSES))
            cv_scores = []
            # In addition, we'll record the predictions for each fold.
            # This will have a 1:1 correspondence with the labels at each fold.
            all_labels = []
            all_predictions = []
            all_raw_predictions = []
            all_urls = []
            i = 0
            for train_index, test_index in group_ksplit:
                model_cv = model_cv_fn(warmstart=warmstart)
                if warmstart and model_name in WARMSTART_MODELS:
                    self.X_train, self.X_test = self.X_warm[train_index], self.X_warm[test_index]
                    self.y_train, self.y_test = self.y_warm[train_index], self.y_warm[test_index]
                    self.imgs_train, self.imgs_test = self.imgs_warm[train_index], self.imgs_warm[test_index]
                    self.X_url_test = self.X_urls_warm[test_index]
                    self.preds = self._train(model_cv, model_name, warmstart=True)
                    if model_name == 'GradientBoostingRegressor':
                        model_cv.n_estimators += 130

                self.X_train, self.X_test = self.X[train_index], self.X[test_index]
                self.y_train, self.y_test = self.y[train_index], self.y[test_index]
                self.imgs_train, self.imgs_test = self.imgs[train_index], self.imgs[test_index]
                self.X_url_test = self.X_urls[test_index]
                self.preds = self._train(model_cv, model_name, warmstart=False)
                
                raw_preds = self.preds
                if self.model_type == 'ngboost':
                    self.preds = np.clip(np.around(self.preds), 0, 5).astype(int) - 1
                    self.y_test = np.clip(np.around(self.y_test), 0, 5).astype(int) - 1
                else:
                    self.preds = np.clip(np.around(self.preds), 0, 4).astype(int)
                    self.y_test = np.clip(np.around(self.y_test), 0, 4).astype(int)
                acc, prec, recall, acc_off_by_one, recall_off_by_one, C_ = self._eval()
                C += C_
                cv_scores.append(acc)
                all_labels.extend(self.y_test)
                all_predictions.extend(self.preds)
                all_urls.extend(self.X_url_test)
                all_raw_predictions.extend(raw_preds)
                print(acc)

            mistakes_str = get_error_urls(all_labels, all_predictions, all_urls, all_raw_predictions)
            kappas_coef = get_kappas_score(all_labels, all_predictions)
            _, overall_prec, overall_recall = acc_prec_recall(C)
            model_report = [model_name, np.mean(cv_scores), overall_prec, overall_recall, acc_off_by_one, recall_off_by_one, C, mistakes_str, model_cv, cv_scores, kappas_coef]
            self._write_model_report(model_report, write_mistakes_str=True)
            # if model_name in ['GradientBoostingRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'NGBRegressor']:
            #     self.plot_shap(model_cv, model_name)


            if model_name == 'GradientBoostingRegressor':
                print("Saving model")
                # Save the current GBT model, trained on the entire dataset
                model_final = model_cv_fn(warmstart=False)
                model_final.fit(self.X, self.y)
                filename = f"{self.model_savepath}/{self.model_savename}"
                joblib.dump(model_final, filename, compress=9)

            # if model_name == 'NGBRegressor':
            #     model_final = model_cv_fn(warmstart=False)
            #     model_final.fit(self.X, self.y)
            #     file_path = Path(f"{self.model_savepath}/{self.model_savename}")
            #     with file_path.open("wb") as f:
            #         pickle.dump(model_final, f)

            mse = np.sum((np.array(all_labels) - np.array(all_raw_predictions)) ** 2)
            print(mse)

            if self.save_results:
                # Save a data structure with the labels, preds, and urls
                results = {
                    "labels": all_labels,
                    "predictions": all_predictions,
                    "urls": all_urls,
                    "raw_preds": all_raw_predictions
                }
                np.savez('./results.npz', labels=results['labels'], predictions=results['predictions'], urls=results['urls'], raw_preds=results['raw_preds'])


    def plot_shap(self, model, model_name):
        feature_list = [fn_dict['features'] for fn_dict in self.dataset_config.proc_fns]
        feature_list = sum(feature_list, [])

        # Compute shapley values from training data and fitted model
        explainer = shap.TreeExplainer(model)
        self.X_train = pd.DataFrame(self.X_train, columns=feature_list)
        print(self.X_train)
        shap_values = explainer.shap_values(np.array(self.X_train), np.array(self.y_train.astype(int)))
        shap.summary_plot(shap_values, self.X_train, show=False, plot_size=(30,10))
        plt.savefig(f"./summary_plot_{model_name}.pdf")


def main(args):
    ml_config = get_config(None, args.train_config, load_only=True)
    dataset_config = get_config(None, args.dataset_config, load_only=True)
    ml_config.ml_params['filename'] = f'ml_{args.ml_model_type}_train_results'
    ml_config.ml_params['model_savename'] = f'ml_{args.ml_model_type}.joblib.pkl'
    ml_config.ml_params['model_type'] = f'{args.ml_model_type}'
    ml_trainer = MLTrainer(args.model_savepath, ml_config, dataset_config, args.warmstart == 'true', args.save_results == 'true')
    ml_trainer.begin_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="For running different kinds of feature-extraction experiments on LAT trays.")
    parser.add_argument('--train_config', type=str, required=True)
    parser.add_argument('--dataset_config', type=str, required=True)
    parser.add_argument('--model_savepath', type=str, required=False)
    parser.add_argument('--ml_model_type', type=str, required=True, help="The type of model (either classifier or regressor)")
    parser.add_argument('--warmstart', type=str, required=True, help="If set to true, model will be trained once, using the warm dataset as a warmstart.")
    parser.add_argument('--save_results', type=str, required=False, help="If set to true, we will save the results of the model.")
    args = parser.parse_args()
    if args.ml_model_type not in ['classifier', 'regressor', 'ngboost']:
        raise Exception("The ml_model_type argument must be from the following list: [classifier, regressor, ngboost]")
    main(args)

