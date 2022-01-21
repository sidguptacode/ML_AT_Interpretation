import numpy as np
import cv2
from well_matrix import Well
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'..')
from constants import MIN_BLOB_AREAS, LARGE_BLOB_AREAS, MIN_PIXEL_CONSIDERATION_AREAS, IMGWS, IMGHS
import scipy.ndimage.filters

"""
    This file defines a set of feature processing functions, and exports
    them all in a dictionary.
"""

def compute_daad(well: Well):
    well_img_flat = well.get_image_alias().flatten()
    well_img_flat = np.sort(well_img_flat)
    # Compute the DAAD score
    x1 = int(0.42*len(well_img_flat))
    x2 = int(0.84*len(well_img_flat))
    delta_x = x2 - x1
    delta_y = well_img_flat[x2] - well_img_flat[x1]
    return delta_y/delta_x


def draw_contours(well: Well):
    contours, hierarchy = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    well_img_2 = cv2.cvtColor(well.well_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Only draw the "big" contours
    MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
    contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
    well.well_image = cv2.drawContours(well_img_2, contours, -1, (0,0,255), -1)
    # Visualizes blob sizes
    if len(contours) != 0:
        well.well_image = cv2.putText(well.well_image, f'{sum([cv2.contourArea(contour) for contour in contours])}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
    return 0


def total_number_of_blobs(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    total_num_all_blobs = len(well.contours)

    # TODO: This plot behaves weird; should investigate.
    return total_num_all_blobs


def number_of_large_blobs(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)

    LARGE_BLOB_AREA = LARGE_BLOB_AREAS[well.type]
    num_large_blobs = 0
    for contour in well.contours:
        if cv2.contourArea(contour) > LARGE_BLOB_AREA:
            num_large_blobs += 1
    return num_large_blobs


def total_area_of_all_blobs(well: Well):
    # all_blobs_pixel_area = np.sum(well.get_image_alias() > 0)
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        return 0
    # Get the largest blob
    all_blobs_pixel_area = sum([cv2.contourArea(contour) for contour in well.contours])
    if well.type == "COVID":
        return all_blobs_pixel_area

    IMGW, IMGH = IMGWS[well.type], IMGHS[well.type]
    well_interior_area = IMGW*IMGH - np.sum(well.exterior_mask == 0)
    area_proportion = all_blobs_pixel_area / well_interior_area
    if area_proportion > 1:
        area_proportion = 1

    return area_proportion


def area_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)

    if len(well.contours) == 0:
        return 0
    # Get the largest blob
    blob_areas = [cv2.contourArea(contour) for contour in well.contours]
    largest_blob_area = max(blob_areas)
    if well.type == "COVID":
        return largest_blob_area

    # Run below code only for the blood experiments
    IMGW, IMGH = IMGWS[well.type], IMGHS[well.type]
    well_interior_area = IMGW*IMGH - np.sum(well.exterior_mask == 0)
    largest_blob_prop = largest_blob_area / well_interior_area
    if largest_blob_prop > 1:
        largest_blob_prop = 1

    return largest_blob_prop


def mean_light_intensity_of_all_blobs(well: Well):
    well_img = well.get_image_alias().copy()
    well_img = well_img.astype('float')

    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    contours, _ = cv2.findContours(binary_well_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
    contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
    blob_area_in_pixels = sum([cv2.contourArea(contour) for contour in contours])
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    # NOTE: Only want to do these for the masked blobs. Also, I think we only care about this feature when the blob is relatively big.
    well_img[well_img == 0] = np.nan
    return np.nanmean(well_img)


def stdev_light_intensity_of_all_blobs(well: Well):
    well_img = well.get_image_alias().copy()
    well_img = well_img.astype('float')
    
    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    contours, _ = cv2.findContours(binary_well_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
    contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
    blob_area_in_pixels = sum([cv2.contourArea(contour) for contour in contours])
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    well_img[well_img == 0] = np.nan
    return np.nanstd(well_img)



def rsd_light_intensity_of_all_blobs(well: Well):
    well_img = well.get_image_alias().copy()
    well_img = well_img.astype('float')
 
    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    contours, _ = cv2.findContours(binary_well_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
    contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
    blob_area_in_pixels = sum([cv2.contourArea(contour) for contour in contours])
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    well_img[well_img == 0] = np.nan
    rsd =  100 * np.nanstd(well_img) / np.nanmean(well_img)
    return rsd


def mean_light_intensity_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    # Get the largest blob
    blob_areas = np.array([cv2.contourArea(contour) for contour in well.contours])
    largest_blob_ind = np.argmax(blob_areas)
    # Draw a mask containing the largest blob
    blob_mask = cv2.drawContours(well.get_image_alias(), well.contours[largest_blob_ind], -1, (1), -1)
    blob_mask = blob_mask != 0
    blob_mask = blob_mask * 1
    # Returning mean light intensity of masked blob
    masked_blob = well.get_image_alias() * blob_mask
    well_img = masked_blob.copy()
    well_img = well_img.astype('float')

    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    well_img[well_img == 0] = np.nan
    return np.nanmean(well_img)



def stdev_light_intensity_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    # Get the largest blob
    blob_areas = np.array([cv2.contourArea(contour) for contour in well.contours])
    largest_blob_ind = np.argmax(blob_areas)
    # Draw a mask containing the largest blob
    blob_mask = cv2.drawContours(well.get_image_alias(), well.contours[largest_blob_ind], -1, (1), -1)
    blob_mask = blob_mask != 0
    blob_mask = blob_mask * 1
    # Returning stdev light intensity of masked blob
    masked_blob = well.get_image_alias() * blob_mask
    well_img = masked_blob.copy()
    well_img = well_img.astype('float')

    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    well_img[well_img == 0] = np.nan
    return np.nanstd(well_img)



def rsd_light_intensity_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    # Get the largest blob
    blob_areas = np.array([cv2.contourArea(contour) for contour in well.contours])
    largest_blob_ind = np.argmax(blob_areas)
    # Draw a mask containing the largest blob
    blob_mask = cv2.drawContours(well.get_image_alias(), well.contours[largest_blob_ind], -1, (1), -1)
    blob_mask = blob_mask != 0
    blob_mask = blob_mask * 1
    # Returning rsd light intensity of masked blob
    masked_blob = well.get_image_alias() * blob_mask
    well_img = masked_blob.copy()
    well_img = well_img.astype('float')

    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    well_img[well_img == 0] = np.nan
    rsd =  100 * np.nanstd(well_img) / np.nanmean(well_img)
    return rsd


def eccentricity_MA_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    # Get the largest blob
    blob_areas = np.array([cv2.contourArea(contour) for contour in well.contours])
    largest_blob_ind = np.argmax(blob_areas)
    # Get it's major axis
    _, (MA,_), _ = cv2.fitEllipse(well.contours[largest_blob_ind])
    return MA


def eccentricity_ma_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    # Get the largest blob
    blob_areas = np.array([cv2.contourArea(contour) for contour in well.contours])
    largest_blob_ind = np.argmax(blob_areas)
    # Get it's minor axis
    _, (_,ma), _ = cv2.fitEllipse(well.contours[largest_blob_ind])
    return ma


def mean_eccentricity_MA_of_all_blobs(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        return 0
    major_axes = np.array([cv2.fitEllipse(contour)[1][0] for contour in well.contours])
    return np.mean(major_axes)


def mean_eccentricity_ma_of_all_blobs(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    minor_axes = np.array([cv2.fitEllipse(contour)[1][1] for contour in well.contours])
    return np.mean(minor_axes)


def stdev_eccentricity_MA_of_all_blobs(well: Well):
    """
        Note: Not used since it doesn't make much sense.
    """
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    major_axes = np.array([cv2.fitEllipse(contour)[1][0] for contour in well.contours])
    return np.std(major_axes)


def stdev_eccentricity_ma_of_all_blobs(well: Well):
    """
        Note: Not used since it doesn't make much sense.
    """
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan
        
    minor_axes = np.array([cv2.fitEllipse(contour)[1][1] for contour in well.contours])
    return np.std(minor_axes)


def rsd_well(well: Well):
    """
        Computes the %RSD of a well.
    """
    well_img = well.get_image_alias().copy()
    well_img = well_img.astype('float')

    well_img[well_img == 0] = np.nan
    rsd =  100 * np.nanstd(well_img) / np.nanmean(well_img)
    return rsd


def total_perimeter_of_all_blobs(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        return 0
    total_perimeter = 0
    for cnt in well.contours:
        total_perimeter += cv2.arcLength(cnt,True)
    return total_perimeter


def perimeter_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        return 0
    # Get the largest blob
    blob_areas = np.array([cv2.contourArea(contour) for contour in well.contours])
    largest_blob_ind = np.argmax(blob_areas)
    # Return it's perimeter
    return cv2.arcLength(well.contours[largest_blob_ind],True)


def centroid_x_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan
    # Get the largest blob
    blob_areas = np.array([cv2.contourArea(contour) for contour in well.contours])
    largest_blob_ind = np.argmax(blob_areas)
    M = cv2.moments(well.contours[largest_blob_ind])
    cx = int(M['m10']/M['m00'])
    return cx


def centroid_y_of_largest_blob(well: Well):
    if well.contours is None:
        contours, _ = cv2.findContours(well.get_image_alias().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_BLOB_AREA = MIN_BLOB_AREAS[well.type]
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_BLOB_AREA]
        well.set_contours(contours)
    if len(well.contours) == 0:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan
    # Get the largest blob
    blob_areas = np.array([cv2.contourArea(contour) for contour in well.contours])
    largest_blob_ind = np.argmax(blob_areas)
    M = cv2.moments(well.contours[largest_blob_ind])
    cy = int(M['m01']/M['m00'])
    return cy


def mean_light_intensity_of_background(well: Well):
    well_img = well.get_image_alias().copy()
    well_img = well_img.astype('float')

    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    well_img[well_img == 0] = np.nan
    return np.nanmean(well_img)


def stdev_light_intensity_of_background(well: Well):
    well_img = well.get_image_alias().copy()
    well_img = well_img.astype('float')

    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    well_img[well_img == 0] = np.nan
    return np.nanstd(well_img)


def rsd_light_intensity_of_background(well: Well):
    well_img = well.get_image_alias().copy()
    well_img = well_img.astype('float')

    binary_well_img = well_img > 0
    blob_area_in_pixels = np.sum(binary_well_img)
    MIN_PIXEL_CONSIDERATION_AREA = MIN_PIXEL_CONSIDERATION_AREAS[well.type]
    if blob_area_in_pixels < MIN_PIXEL_CONSIDERATION_AREA:
        # In this case, there are no blobs, so we will impute this value with it's feature mean.
        return np.nan

    well_img[well_img == 0] = np.nan
    rsd =  100 * np.nanstd(well_img) / np.nanmean(well_img)
    return rsd


def rsd_radial_dist_from_center(well: Well):
    """
        Computes the nanmean of the image. Then, slides a convolutional
        filter and computes the standard deviation of that filter w.r.t the image nanmean.
        Finds the maximum window location, and returns it's radial distance from the center.
        NOTE: This feature may be useless, but we can see through accuracies / SHAP plots.
        It's mainly to get a sense of the "location" of specs / blobs.
        NOTE: This feature is not used, because it's too time expensive!
    """
    well_img = well.get_image_alias().copy().astype(float)
    well_img[well_img == 0] = np.nan
    k = 10
    img_mean = np.nanmean(well_img)
    def filter_fn(window, k, img_mean):
        window = window.reshape((k,k))
        if np.isnan(np.sum(window)):
            return -1
        window -= img_mean
        window = window ** 2
        window_stdev = np.sqrt(np.sum(window) / (k ** 2))
        return window_stdev
    stdev_matrix = scipy.ndimage.filters.generic_filter(well_img,lambda window: filter_fn(window, k, img_mean),footprint=np.ones((k,k)),mode='constant',cval=0.0)
    assert stdev_matrix.shape == well_img.shape
    max_stdev_coords = np.unravel_index(np.argmax(stdev_matrix, axis=None), stdev_matrix.shape)
    center = (91, 91)
    radial_dist = np.sqrt(np.sum([(center[i] - max_stdev_coords[i]) ** 2 for i in range(2)]))
    return radial_dist


COMPUTE_FEATURE_FNS = {
    'frame_num': lambda well: well.frame,
    'draw_contours': lambda well: draw_contours(well),
    'total_number_of_blobs': lambda well: total_number_of_blobs(well),
    'number_of_large_blobs': lambda well: number_of_large_blobs(well),
    'total_area_of_all_blobs': lambda well: total_area_of_all_blobs(well),
    'area_of_largest_blob': lambda well: area_of_largest_blob(well),
    'mean_light_intensity_of_all_blobs': lambda well: mean_light_intensity_of_all_blobs(well),
    'stdev_light_intensity_of_all_blobs': lambda well: stdev_light_intensity_of_all_blobs(well),
    'rsd_light_intensity_of_all_blobs': lambda well: rsd_light_intensity_of_all_blobs(well),
    'mean_light_intensity_of_largest_blob': lambda well: mean_light_intensity_of_largest_blob(well),
    'stdev_light_intensity_of_largest_blob': lambda well: stdev_light_intensity_of_largest_blob(well),
    'rsd_light_intensity_of_largest_blob': lambda well: rsd_light_intensity_of_largest_blob(well),
    'eccentricity_MA_of_largest_blob': lambda well: eccentricity_MA_of_largest_blob(well),
    'eccentricity_ma_of_largest_blob': lambda well: eccentricity_ma_of_largest_blob(well),
    'mean_eccentricity_MA_of_all_blobs': lambda well: mean_eccentricity_MA_of_all_blobs(well),
    'mean_eccentricity_ma_of_all_blobs': lambda well: mean_eccentricity_ma_of_all_blobs(well),
    'stdev_eccentricity_MA_of_all_blobs': lambda well: stdev_eccentricity_MA_of_all_blobs(well),
    'stdev_eccentricity_ma_of_all_blobs': lambda well: stdev_eccentricity_ma_of_all_blobs(well),
    'rsd': lambda well: rsd_well(well),
    'total_perimeter_of_all_blobs': lambda well: total_perimeter_of_all_blobs(well),
    'perimeter_of_largest_blob': lambda well: perimeter_of_largest_blob(well),
    'centroid_x_of_largest_blob': lambda well: centroid_x_of_largest_blob(well),
    'centroid_y_of_largest_blob': lambda well: centroid_y_of_largest_blob(well),
    'mean_light_intensity_of_background': lambda well: mean_light_intensity_of_background(well),
    'stdev_light_intensity_of_background': lambda well: stdev_light_intensity_of_background(well),
    'rsd_light_intensity_of_background': lambda well: rsd_light_intensity_of_background(well),
    'compute_daad': lambda well: compute_daad(well),
    'rsd_radial_dist_from_center': lambda well: rsd_radial_dist_from_center(well)
}

def get_feature_ind_dict(feature_list):
    feature_ind_dict = {}
    for i, feature in enumerate(feature_list):
        feature_ind_dict[feature] = i
    feature_ind_dict['ml_agg_score'] = -2
    feature_ind_dict['ground_truth'] = -1
    return feature_ind_dict