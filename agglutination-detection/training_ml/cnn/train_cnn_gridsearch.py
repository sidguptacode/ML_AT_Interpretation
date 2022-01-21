from operator import mod
from scipy.stats import ttest_rel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'..')
from constants import NUM_CLASSES
from helpers import get_config
from cnn_model import cnn_model, setup_tensorboard, get_all_combs
from custom_loss_mlp import custom_mlp_model
# import shap as shap
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict
from training_helpers import get_all_models_dict, acc_prec_recall, acc_recall_off_by_one, get_error_urls
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from hyperparameters import HP_FINAL_OUTPUT_CHANNEL_SIZE, HP_DENSE_LAYER_SIZES, METRIC_ACCURACY

EPS = 0.001

class MLTrainer():

    def __init__(self, model_savepath, ml_config, dataset_config, num_classes=4) -> None:
        self.model_savepath = model_savepath
        self.ml_params = ml_config.ml_params
        self.num_classes = num_classes
        self.dataset_config = dataset_config
        self.seed = ml_config.seed
        self.model_savename = self.ml_params['model_savename']
        self.model_type = self.ml_params['model_type']


    def _train_and_test(self, model, keras_model=False):
        if keras_model:
            model.fit(self.imgs_train, self.y_train, batch_size=20, epochs=1, callbacks=self.callbacks)
            preds = model.predict(self.imgs_test)
        else:
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
        return preds
        

    def _eval(self, y_test, preds):
        C = confusion_matrix(y_test, preds)
        acc, prec, recall = acc_prec_recall(C)
        acc_off_by_one, recall_off_by_one = acc_recall_off_by_one(C)
        return acc, prec, recall, acc_off_by_one, recall_off_by_one, C


    def _write_model_report(self, model_report, write_mistakes_str=False):
        model_name, accuracy, precision, recall, acc_off_by_one, recall_off_by_one, conf_matrix, mistakes_str, cv_scores, model = model_report
        with open(f"./ml_regressor_report.txt", "a") as outf:
            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {model_name}:\n')  
            outf.write(f'\tAccuracy: {accuracy:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision]}\n')
            outf.write(f'\tAccuracy discounting off-by-one errors: {acc_off_by_one:.4f}\n')
            outf.write(f'\tRecall discounting off-by-one errors: {[round(item, 4) for item in recall_off_by_one]}\n')
            outf.write(f'\tConfusion Matrix: \n{ conf_matrix.astype(int)}\n\n')
            outf.write(f'\tCV Scores: \n{[round(item, 4) for item in cv_scores]}\n\n')
            outf.write(f'\tCV Mean: \n{np.mean(cv_scores):.4f}\n\n')
            # outf.write(f'\t{model.summary()}\n\n')
            if model_name == 'CNN':
                model.summary(print_fn=lambda x: outf.write(x + '\n'))
                # outf.write(f'\tHyperparams \n {hyperparams}\n\n')
        if write_mistakes_str:
            with open(f"./mistake_urls/ml_regressor_report_{model_name}_mistakes.txt", "w") as outf:
                outf.write(f'\t{mistakes_str}\n\n')


    def begin_training(self) -> None:
        """
            Loads in the datasets, and runs the ML experiments.
        """
        # Precondition: The dataset is normalized
        dataset = np.load('./dataset.npz')
        self.X, self.y, self.groups = dataset['X'], dataset['y'], dataset['groups']
        self.imgs = dataset['imgs']
        self.X_urls = dataset['X_urls']
        self.imgs = np.expand_dims(self.imgs, axis=-1) / 255

        if self.ml_params['model_type'] == 'classifier':
            self.y = np.around(self.y)
        self.num_classes = len(np.unique(np.around(self.y)))
        self.run_ml_experiments(None)


    def run_cross_val_experiments(self, model_name, model_instantiator, write_mistakes_str=False):
        # Create the cross-validation split
        group_kfold = GroupKFold(n_splits=3)
        group_ksplit = group_kfold.split(self.X, self.y, self.groups)
        # We'll have a confusion matrix which gets accumulated, and we'll record accuracy at each fold.
        C = np.zeros((NUM_CLASSES, NUM_CLASSES))
        cv_scores = []
        # In addition, we'll record the predictions for each fold.
        # This will have a 1:1 correspondence with the labels at each fold.
        all_labels = []
        all_predictions = []
        all_urls = []
        for train_index, test_index in group_ksplit:
            model = model_instantiator()
            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.imgs_train, self.imgs_test = self.imgs[train_index], self.imgs[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]
            self.X_url_test = self.X_urls[test_index]
            self.preds = self._train_and_test(model, keras_model=model_name=='CNN')
            self.preds = np.clip(np.around(self.preds), 0, 4).astype(int)
            self.y_test = np.clip(np.around(self.y_test), 0, 4).astype(int)
            acc, prec, recall, acc_off_by_one, recall_off_by_one, C_ = self._eval(self.y_test, self.preds)
            C += C_
            cv_scores.append(acc)
            all_labels.extend(self.y_test)
            all_predictions.extend(self.preds)
            all_urls.extend(self.X_url_test)
        mistakes_str = get_error_urls(all_labels, all_predictions, all_urls) if write_mistakes_str else "Not computing mistakes_str"
        model_report = [model_name, np.mean(cv_scores), prec, recall, acc_off_by_one, recall_off_by_one, C, mistakes_str, cv_scores, model]
        return model_report


    def gridsearch_cnn(self):
        # Train a series of CNNs in a grid-search, and store the results in tensorboard.
        # Set up a place to write our grid search results to tensorboard logs
        grid_file_path = setup_tensorboard()
        # Get a list of hyperparameter configurations.
        all_combs = get_all_combs()
        # Begin grid search
        for comb in all_combs:
            hparams = {
                HP_FINAL_OUTPUT_CHANNEL_SIZE: comb['output_channel_size'],
                HP_DENSE_LAYER_SIZES: str(comb['dense_layer_sizes']),
            }
            # Creata a lambda function which constructs a CNN from these hyperparams
            model_instantiator = lambda: cnn_model(hparams)
            # Create the tensorboard logs for this configuration
            dense_sizes_str = "_".join([str(size) for size in comb['dense_layer_sizes']])
            curr_run_name = f"{grid_file_path}/{hparams[HP_FINAL_OUTPUT_CHANNEL_SIZE]}outchansize_{dense_sizes_str}densesizes"
            curr_run_writer = tf.summary.create_file_writer(curr_run_name)
            with curr_run_writer.as_default():
            # curr_run_writer = tf.summary.FileWriter(curr_run_name)
            # with curr_run_writer:
                # Write the hyperparams
                hp.hparams(hparams)
                # Callbacks for our CNN
                self.callbacks = [tf.keras.callbacks.TensorBoard(log_dir=curr_run_name, histogram_freq=1, profile_batch=0)]
                # Train and get the results
                model_report = self.run_cross_val_experiments('CNN', model_instantiator, write_mistakes_str=False)
                # Store some results locally (confusion matrix, accuracy, etc)
                self._write_model_report(model_report, write_mistakes_str=False)
                # Log some results to tensorboard
                _, cv_mean_acc, prec, recall, acc_off_by_one, recall_off_by_one, _, _, _, _ = model_report
                # Logging custom scalars to Tensorboard
                tf.compat.v1.summary.scalar(METRIC_ACCURACY, cv_mean_acc)
                # summary = tf.Summary()
                # summary.value.add(tag=METRIC_ACCURACY, simple_value=cv_mean_acc)
                # curr_run_writer.add_summary(summary, 1)


    def run_ml_experiments(self, hyperparams=None):
        ''' This function performs the ML experiments.'''
        self.all_models_cv = get_all_models_dict(self.ml_params)
        for model_name in self.all_models_cv:
            print(f"Training {model_name} using cross validation")
            if model_name == 'CNN':
                self.gridsearch_cnn()
            else:
                model_instantiator = self.all_models_cv[model_name]
                model_report = self.run_cross_val_experiments(model_name, model_instantiator, write_mistakes_str=False)
                self._write_model_report(model_report, write_mistakes_str=False)


    # def plot_shap(self, model):
    #     feature_list = [fn_dict['features'] for fn_dict in self.dataset_config.proc_fns]
    #     feature_list = sum(feature_list, [])

    #     # Compute shapley values from training data and fitted model
    #     explainer = shap.TreeExplainer(model)
    #     self.X_train = pd.DataFrame(self.X_train, columns=feature_list)
    #     print(self.X_train)
    #     shap_values = explainer.shap_values(np.array(self.X_train), np.array(self.y_train.astype(int)))
    #     shap.summary_plot(shap_values, self.X_train, show=False, plot_size=(30,10))
    #     plt.savefig("./summary_plot.pdf")


def main(args):
    ml_config = get_config(None, args.train_config, load_only=True)
    dataset_config = get_config(None, args.dataset_config, load_only=True)
    ml_config.ml_params['filename'] = f'ml_{args.ml_model_type}_train_results'
    ml_config.ml_params['model_savename'] = f'ml_{args.ml_model_type}.joblib.pkl'
    ml_config.ml_params['model_type'] = f'{args.ml_model_type}'
    ml_trainer = MLTrainer(args.model_savepath, ml_config, dataset_config)
    ml_trainer.begin_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="For running different kinds of feature-extraction experiments on LAT trays.")
    parser.add_argument('--train_config', type=str, required=True)
    parser.add_argument('--dataset_config', type=str, required=True)
    parser.add_argument('--model_savepath', type=str, required=False)
    parser.add_argument('--ml_model_type', type=str, required=True, help="The type of model (either classifier or regressor)")
    args = parser.parse_args()
    if args.ml_model_type not in ['classifier', 'regressor']:
        raise Exception("The ml_model_type argument must be from the following list: [classifier, regressor]")
    main(args)

