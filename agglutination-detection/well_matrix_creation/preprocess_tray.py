import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from itertools import product
import pandas as pd
from well_matrix import Well
import datetime
from datetime import datetime
import os
from sklearn.cluster import KMeans
from collections import Counter


def crop_to_plate(image):
    """
      Crops only the desired plate, and no surrounding black regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (5,5))
    thresh = cv2.threshold(blur, int(gray.max()*0.7), 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnts[-1])
    
    return image[y:y+h, x:x+w].copy()


def rotate_n_flip(image, rot=cv2.ROTATE_90_CLOCKWISE):
    """
      Rotates 90 degrees and flips.
    """
    rotated = cv2.rotate(image, rot)
    return cv2.flip(rotated, 0)


def cor_lum(image):
    """
      Applies localized histogram equalization to each of the image color channels.
    """
    b,g,r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(50,50))
    return cv2.merge([clahe.apply(c) for c in (b,g,r)])


def cut_arr(arr, step):
    """
      Used when re-sorting the array of circle centers.
      Parameters:
        arr: Array of (x, y) circle centers
        step: radius, in this case.
    """
    x, y = arr[0]
    # Index all y-positions before this radius
    idx = arr[: ,1]<(y+step)
    # First term: Second term: Truncated array
    return arr[idx], arr[~idx]

def contour_center(contour):
    """
      Gets the center of mass of the contour (I believe).
      This function is not used.
    """
    M = cv2.moments(contour)
    if  M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return 0, 0


def read_wells(image, circles, radius=91, erode=30, debug=False, map_as_str=False):
    """
      Given the assay image and a set of circles representing each well,
      visualize each circle, and compute an agglutination score.
    """
    mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (radius - erode), 2 * (radius - erode)))
    mask = cv2.copyMakeBorder(mask, *[erode] * 4, 0)

    h, w = image.shape[:2]
    dims = np.array((w, h))

    circles = [cv2.minEnclosingCircle(cnt)[0] for cnt in circles]
    circles = np.asarray(circles).astype('int')

    circles = circles[circles[:, 1].argsort(kind='mergesort')]

    trunc_circles = circles.copy()
    new_cirlces = []
    cut_arr(trunc_circles, radius)
    while len(trunc_circles) > 1:
        cut, trunc_circles = cut_arr(trunc_circles, radius)
        cut = cut[cut[:, 0].argsort(kind='mergesort')]
        new_cirlces.append(cut)
    circles = np.concatenate(new_cirlces)

    if map_as_str:
        samps = {}
    else:
        samps = []
    mapper = []

    if debug:
        cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    i = 0
    for center in circles:
        if (((center - radius) < 0) | ((center + radius) > dims)).any():
            if debug:
                print(center, radius, center - radius * 2, center + radius * 2)
                cv2.circle(cimg, (center[0], center[1]), radius, (255, 0, 0), 10)
            continue
        if debug:
            cv2.circle(cimg, (center[0], center[1]), radius, (0, 255, 0), 10)

        i += 1
        x, y = center.ravel()
        temp_img = image[y - radius:y + radius, x - radius:x + radius].copy()
        # temp_img[mask == 0] = 0
        well_sample = temp_img

        if map_as_str:
            samps[i] = well_sample
            mapper.append("{}:({},{})".format(i, x, y))
        else:
            samps.append(well_sample)
            mapper.append((x, y))
    if debug:
        plt.imshow(cimg)
        plt.show()
    mapper = np.array(mapper)
    return samps, mapper


def find_wells(image):
    """
      Given a grayscale image of the assay, locate the wells with hough circles.
    """
    gray = image.copy()
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 10.0, minDist=130,
                               param1=110, param2=0.5,
                               minRadius=70,
                               maxRadius=120)
    if circles is not None:
        # Create a mask that stores the center of each detected circle.
        mask = np.zeros(gray.shape, dtype='uint8')
        for c in circles:
            c = c.ravel().astype('int')
            cv2.circle(mask, (c[0], c[1]), 20, (255), -1)
        # Now, we find the contours for each circle in the mask.
        # These contours are the zero'th output value, and are represented as an array of points.
        return cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    else:
        return None


def map_columns(df, custom_map=None):
    if custom_map is None:
        custom_map = {i + 1: '{}{}'.format(*j) for i, j in
                      enumerate(list(product(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], range(1, 13))))}
    return [custom_map.get(col, i) for i, col in enumerate(df.columns)]

