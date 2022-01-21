from __future__ import print_function
from functools import partial
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import os
from hyperparameters import HP_FINAL_OUTPUT_CHANNEL_SIZE, HP_DENSE_LAYER_SIZES, METRIC_ACCURACY
import ast


def get_all_combs():
    """ 
        Returns all combinations of hyperparameters, based off of the constants defined above.
    """
    all_combs = []
    for output_channel_size in HP_FINAL_OUTPUT_CHANNEL_SIZE.domain.values:
        for dense_layer_sizes in HP_DENSE_LAYER_SIZES.domain.values:
            config = {
                'output_channel_size': output_channel_size, 
                'dense_layer_sizes': ast.literal_eval(dense_layer_sizes)
            }
            if config not in all_combs:
                all_combs.append(config)
    return all_combs


def setup_tensorboard():
    # Get folder paths for tensorboard logs
    folder_head = os.path.dirname('./tensorboard')
    logs_folder = os.path.join(folder_head, 'logs')
    log_folder = os.path.join(logs_folder, 'cnn')
    grid_file_path = f"{log_folder}/hparam_searching"
    setup_hyperparam_table(log_folder, tf.summary.create_file_writer(grid_file_path))
    # setup_hyperparam_table(log_folder, tf.summary.FileWriter(grid_file_path))
    return grid_file_path


def setup_hyperparam_table(log_folder, grid_file_writer):
    """ 
        Sets up hyperparams for tensorboard.
    """
    with grid_file_writer.as_default():
    # with grid_file_writer:
        hp.hparams_config(
            hparams=[HP_FINAL_OUTPUT_CHANNEL_SIZE, HP_DENSE_LAYER_SIZES],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )


def cnn_model(hparams):
    """
        Creates the a model based off of the provided hyperparameters.
    """
    HP_FINAL_OUTPUT_CHANNEL_SIZE, HP_DENSE_LAYER_SIZES = hparams.keys()
    with tf.name_scope('lat_cnn'):
        inputs = Input(shape=(182,182,1), name=f'input_1')

        # Setup layer params
        conv = partial(Conv2D, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')
        maxpool = partial(MaxPooling2D, pool_size=(2, 2), strides=(2, 2), padding='same')
        dense = partial(Dense, activation='relu', kernel_regularizer=l2(0.01))
        final_dense = partial(Dense, activation='relu', kernel_regularizer=l2(0.01))

        output_channel_size = hparams[HP_FINAL_OUTPUT_CHANNEL_SIZE]
        dense_layer_sizes = ast.literal_eval(hparams[HP_DENSE_LAYER_SIZES])

        channel_sizes = []
        while output_channel_size >= 4:
            channel_sizes.append(output_channel_size)
            output_channel_size = output_channel_size // 2

        channel_sizes.reverse()
        # We'll fix the CNNs to have 5 layers
        x = inputs
        for i in range(5):
            x = conv(channel_sizes[i], name=f'conv{i+1}')(x)
            x = maxpool(name=f'pool{i+1}')(x)

        x = Flatten(name=f'flatten')(x)

        for i, dense_size in enumerate(dense_layer_sizes):
            x = dense(dense_size, name=f'fc{i+1}')(x)
        x = final_dense(1, name=f"fc{len(dense_layer_sizes)+1}")(x)

        # Create model
        model = Model(inputs, x, name=f'cnn_model')
        model.compile(optimizer="Adam", loss="mse",  metrics=["mae"])

    return model
