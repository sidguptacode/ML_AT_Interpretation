from __future__ import print_function
import argparse
import numpy as np
import sys
import os
sys.path.insert(0,'..')
from training_ml.training_helpers import get_normalization_constants
from experiment_runner import ExperimentRunner
from download_owncloud_tray_imgs import download_from_oc
from helpers import get_config
from training_ml.pull_labels import pull_labels
from well_matrix import Well
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from training_helpers import acc_prec_recall, acc_recall_off_by_one
from train_cross_validation import compute_confusion_matrix

def main():
   # After training, get the labels, predictions, and the well_url for that data sample.
   results = np.load('./results.npz')
   labels = results['labels']
   preds = results['predictions']
   urls = results['urls']

   # Sort the well_urls as strings
   url_sort_ind = np.argsort(urls)
   urls = urls[url_sort_ind]

   # To do the smoothing, for each unique well, we need to order it's agglutination score 
   # by frame number, and then smooth those values. well_curves will have the following representation:
   # well_curves = { 'unique_well': {'labels' : [ORDERED_LABELS], 'predictions': [ORDERED_PREDICTIONS] }, ... }
   # where the order comes from the frame number
   well_curves = {}

   for i in range(len(urls)):
      url = urls[i]
      url_components = url.split('_')
      
      # Get the well_id, without the frame number
      unique_well = '_'.join(url_components[:-1])
      
      # Get the frame number, while ignoring the '.png'. Cast the frame number as an int.
      frame_num = int(url_components[-1].split('.')[0])

      if unique_well not in well_curves.keys():
         # We use np.ones(400), because no tray runs for more than 400 frames.
         # We multiply by -1, since a label/prediction of -1 denotes a None value.
         well_curves[unique_well] = {'labels': np.ones(400)*(-1), 'predictions': np.ones(400)*(-1)}
      
      well_curves[unique_well]['labels'][frame_num] = labels[i]
      well_curves[unique_well]['predictions'][frame_num] = preds[i]
      

   labels_smoothed = []
   preds_smoothed = [] 
   for unique_well in well_curves:
      well_labels, well_preds = well_curves[unique_well]['labels'], well_curves[unique_well]['predictions']
      
      # Get the x-coordinates of the `non -1` labels.
      non_neg_x = np.where(well_labels != -1)[0]
      # Now remove all the -1 labels from each list
      well_labels = well_labels[non_neg_x]
      well_preds = well_preds[non_neg_x]
      
      num_well_labels = len(well_labels)
      if num_well_labels <= 8:
         # Can't do smoothing without at least 5 labels
         well_labels_smooth = well_labels
         well_preds_smooth = well_preds
      else:
         # Apply smoothing
         # TODO: Apply smoothing for the predictions with predictions of the other wells.
         window_len = 9
         poly_order = 2
         # if num_well_labels < window_len:
         #    # Set the window_len to be the closest odd number to num_well_labels
         #    if num_well_labels % 2 == 0:
         #       window_len = num_well_labels - 1
         #    else:
         #       window_len = num_well_labels - 2

         # if poly_order <= window_len:
         #    poly_order = window_len - 1

         well_labels_smooth = savgol_filter(well_labels, window_len, poly_order) 
         well_preds_smooth = savgol_filter(well_preds, window_len, poly_order)
         plt.clf()
         plt.figure(figsize=(10,10))
         plt.subplot(2, 1, 1)
         plt.plot(non_neg_x, well_labels, label='Ground truth')
         plt.plot(non_neg_x, well_preds, label='Prediction')
         plt.scatter(non_neg_x, well_labels, label='Ground truth')
         plt.scatter(non_neg_x, well_preds, label='Prediction')
         plt.legend()
         plt.rcParams.update({'font.size': 10})
         # plt.title(f"In {image_folder_name}, for group {group}, well {i}, with concentration {conc}")

         plt.subplot(2, 1, 2)
         plt.plot(non_neg_x, well_labels_smooth, label='Ground truth')
         plt.plot(non_neg_x, well_preds_smooth, label='Prediction')
         plt.scatter(non_neg_x, well_labels_smooth, label='Ground truth')
         plt.scatter(non_neg_x, well_preds_smooth, label='Prediction')
         plt.legend()
         plt.rcParams.update({'font.size': 10})
         plt.show()
         # assert False
         # plt.title(f"In {image_folder_name}, for group {group}, well {i}, with concentration {conc}, SMOOTHED")
         # plt.savefig(experiment_dir + f'/{image_folder_name}_group_{group}_well_{i}_conc_{conc}.png')

      labels_smoothed.extend(well_labels_smooth)
      preds_smoothed.extend(well_preds_smooth)

   labels_smoothed = np.clip(np.around(np.array(labels_smoothed)), 0, 4).astype(int)
   preds_smoothed = np.clip(np.around(np.array(preds_smoothed)), 0, 4).astype(int)

   C = compute_confusion_matrix(labels_smoothed, preds_smoothed)
   acc, prec, recall = acc_prec_recall(C)
   acc_off_by_one, recall_off_by_one = acc_recall_off_by_one(C)
   print("\n\n")
   print(C)
   print(f"\nacc: {acc} \n recall: {recall} \n")
   print(f"acc_off_by_one: {acc_off_by_one} \nrecall_off_by_one {recall_off_by_one}")

if __name__ == '__main__':
    main()