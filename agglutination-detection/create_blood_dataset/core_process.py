import cv2
import json
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from blood_typing import *
import sys
sys.path.insert(0,'..')
sys.path.insert(0,f'..{os.sep}..')

CONTOUR_SHRINK = 0.8275 # Optimized as a hyperparameter for the 567 blood dataset
OFFSET=0

def compute_imgs_dict(load):
    perc = 0.20
    table = pd.read_excel('Quad_opt_vals_v2.xlsx', index_col=[0, 1])
    low, high = table.idxmax()['BW']
    mid = None
    cut_off = 0.1512
    plot_results_ = True
    image_dict = {}
    image_dict_flat = {}
    image_dict_crop = {}
    samples_dict = {}
    folder = f'.{os.sep}data{os.sep}Samples{os.sep}'
    # folder = 'samples\\'
    enum = 0
    imgs_dict = {}
    exterior_masks_dict = {}
    if load:
        imgs_dict = np.load('data.npy', allow_pickle=True)[()]
        exterior_masks_dict = np.load('exterior_masks.npy', allow_pickle=True)[()]
        return imgs_dict, exterior_masks_dict

    with open('IB0_CCBR_img_cords.json', 'r') as file:
        img_cords_IB0 = json.load(file)
    with open('IB0_CCBR_ref_cords.json', 'r') as file:
        ref_cords_IB0 = json.load(file)

    with open('MRBOX_img_cords.json', 'r') as file:
        img_cords_MRBOX = json.load(file)
    with open('MRBOX_ref_cords.json', 'r') as file:
        ref_cords_MRBOX = json.load(file)

    for filename in tqdm(glob(folder + '*.jpg')):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        image_name = filename.split(os.sep)[-1].strip('.jpg')

        image_dict[image_name] = img

        if 'RL' in image_name:
            invert = True
        else:
            invert = False

        if ('Standard' in image_name) or ('cross' in image_name):
            positive = True
        else:
            positive = False

        if 'SBK' in image_name:
            negative = True
            invert = True
        else:
            negative = False

        if 'MRBOX' in image_name or 'SBK' in image_name:
            img_cor = perspective_correction(img, img_cords_MRBOX, ref_cords_MRBOX)
            crop_img, sample_list = annotate_image(img_cor, [500, 1025, 385, 2260],
                                                positive=positive, negative=negative,
                                                invert=invert)
        elif 'IB0' in image_name:
            img_cor = perspective_correction(img, img_cords_IB0, ref_cords_IB0, cropped=True)
            crop_img, sample_list = annotate_image(img_cor, [200, 75, 165, 700],
                                                positive=positive, negative=negative,
                                                invert=invert)

        else:
            img_cor = perspective_correction(img)
            crop_img, sample_list = annotate_image(img_cor,
                                                positive=positive, negative=negative,
                                                invert=invert)

        image_dict_flat[image_name] = img_cor
        image_dict_crop[image_name] = crop_img
        samples_dict[image_name] = sample_list

    all_samples_results = {}

    for image in tqdm(image_dict_crop):
        report_quant = None
        result_quant = None
        report_hist = None
        result_hist = None

        img = image_dict_crop[image]

        contours, bounding_boxes = detect_droplets(img)

        if len(contours) < len(samples_dict[image]):
            contours, bounding_boxes = detect_droplets(img, var_enchance=True)

        if len(contours) < len(samples_dict[image]):
            contours, bounding_boxes = detect_droplets(img, var_enchance=True, plot=True)[:2]

        drop_imgs = get_droplets(img, contours, bounding_boxes, samples_dict[image],
                                Offset=OFFSET, wb_cor=True, masked=True, contour_shrink=CONTOUR_SHRINK)
        # use masked droplets
        drop_img = drop_imgs[1]

        try:
            df_pixel = get_pixels(samples_dict[image], images=drop_img, masked=True)

            df_mean, df_std, df_range, df_D = measure_pixels(df_pixel, samples_dict[image],
                                                            normalize=True, strech=True,
                                                            lower=low, mid=mid, higher=high)

            report_quant, result_quant = get_report_quant(df_D,
                                                        channel='BW',
                                                        cut_off=cut_off)

            # Store all results in a dictionary
            all_samples_results[image] = report_quant
        except Exception as e:
            print(image)
            print(e)

        # Show unmasked droplets
        drop_img = drop_imgs[1] # This is the masked image
        # drop_img = drop_imgs[0] # This is the unmasked one
        exterior_masks = drop_imgs[2]
        
        directory = f'..{os.sep}agglutination-data-storage{os.sep}indiv_well_imgs' + os.sep
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        img_name = "bloodversion2" + image.replace(' ', '').replace('-', '_')
        export = directory + img_name

        indiv_imgs, indiv_exterior_masks, sample_list = get_indiv_images(images=drop_img, exterior_masks=exterior_masks, result=result_quant,
                                        image_name=img_name, plot=plot_results_, fscale=5, Offset=OFFSET)

        for i, result_image in enumerate(indiv_imgs):
            imgs_dict[f"{img_name}_{sample_list[i]}.jpg"] = result_image
            exterior_masks_dict[f"{img_name}_{sample_list[i]}.jpg"] = indiv_exterior_masks[i]

    imgs_dict = np.load('data.npy', allow_pickle=True)[()]
    exterior_masks_dict = np.load('exterior_masks.npy', allow_pickle=True)[()]
    return imgs_dict, exterior_masks_dict