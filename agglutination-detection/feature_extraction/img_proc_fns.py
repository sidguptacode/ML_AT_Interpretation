import numpy as np
import cv2
from skimage.filters import threshold_local, threshold_otsu
import matplotlib.pyplot as plt

"""
    This file defines a set of image processing functions, and exports
    them all in a dictionary.
"""


def apply_joint_thresholding(well_img: np.ndarray, hyperparams: dict) -> None:
    """
        Applies both global and adaptive thresholding to an image.
    """
    if hyperparams['global_thresh'] == 'otsu':
        global_thresh = threshold_otsu(well_img)
    else:
        global_thresh = hyperparams['global_thresh']
    well_img_global = well_img < global_thresh
    block_size, offset = hyperparams['block_size'], hyperparams['offset']
    local_thresh = threshold_local(well_img, block_size, offset=offset) - offset
    well_img_local = well_img < local_thresh
    well_img[:, :] = well_img_global + well_img_local


def apply_global_thresholding(well_img: np.ndarray, hyperparams: dict) -> None:
    """
        Transforms the image into a binary representation using Otsu's global thresholding.
        That is, all pixels in the image get labelled as 0 or 1.
    """
    if hyperparams['thresh'] == 'otsu':
        thresh = threshold_otsu(well_img)
    else:
        thresh = hyperparams['thresh']
    well_img[:, :] = well_img < thresh


def apply_adaptive_thresholding(well_img: np.ndarray, block_size: int, offset: int) -> None:
    """
        Reads hyperparameters from the provided .yml config file (block_size & theshold)
        and transforms the image into a binary representation using adaptive thresholding.
        That is, all pixels in the image get labelled as 0 or 1.
    """
    local_thresh = threshold_local(well_img, block_size, offset=offset) - offset
    well_img[:, :] = well_img < local_thresh


def apply_morph(well_img: np.ndarray, kernel_size: int, morph_type: str = 'open') -> None:
    """
        Applies a morphological operation on the image (either morph open or morph close).
        This is used to remove excess noise in the image.
    """
    if morph_type == 'open':
        morph = cv2.MORPH_OPEN
    else:
        morph = cv2.MORPH_CLOSE
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    well_img[:, :] = cv2.morphologyEx(well_img, morph, kernel)


def add_well_boundaries(well_img: np.ndarray, radius: int, thickness: int, color: str = 'black') -> None:
    """ 
        Starting from the center of the well, draw a circle that represents the well boundary.
        This is used to add an artifical black boundary covering the everything except the well
        interior, so that binarizing the image becomes easier. It's also used to add a small white 
        boundary to help with isolating the interior.
    """
    if color == 'white':
        color_tup = (1, 1, 1)
    else:
        color_tup = (0, 0, 0)
    well_center = (well_img.shape[0] // 2, well_img.shape[1] // 2)
    cv2.circle(well_img , well_center, radius, color_tup, thickness)


def remove_artificial_well_boundary(well_img: np.ndarray, color: str) -> None:
    """ 
        Removes an artificial 'white well boundary' that surrounds a well interior.
        If we treat the white pixels as 'land' and all black pixels as 'water',
        we effectively color all pixels that are on the same 'island' as the top left pixel, and the top right pixel.
    """
    img_h, img_w = well_img.shape
    color_val = 1 if color == 'white' else 0

    floodfill_color = (not color_val) * 1
    cv2.floodFill(well_img, None, (0, 0), floodfill_color)
    cv2.floodFill(well_img, None, (img_h - 1, 0), floodfill_color)
    cv2.floodFill(well_img, None, (0, img_w - 1), floodfill_color)
    cv2.floodFill(well_img, None, (img_h - 1, img_w - 1), floodfill_color)


def mask_binary_onto_original(well_img: np.ndarray, well_saved_states: np.ndarray) -> None:
    """
        Mask the original well with this processed binarized well.
    """
    orig_well_img = well_saved_states['original']
    well_img_mask = well_saved_states['remove_artificial_well_boundary']
    well_img[:, :] = np.multiply(well_img_mask, orig_well_img)


def mask_binary_inv_onto_original(well_img: np.ndarray, well_saved_states: np.ndarray) -> None:
    """ 
        Masks the inverse of the binary onto the original. This has the effect of keeping only the background,
        and removing the blobs. 
    """
    orig_well_img = well_saved_states['original']
    well_img_mask = well_saved_states['apply_morph_close']
    well_img_mask_inv = (-1 * well_img_mask) + 1
    well_img[:, :] = np.multiply(well_img_mask_inv, orig_well_img)


def apply_clahe(well_img: np.ndarray, clip_limit, tile_grid_size):
    """
        Standardizes the image contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl1 = clahe.apply(well_img)
    well_img[:, :] = cl1[:, :]


IMG_PROC_FNS = {
    'original': lambda well_img, _: None,
    'apply_clahe': lambda well_img, hyperparams: apply_clahe(well_img, hyperparams['clip_limit'], hyperparams['tile_grid_size']),
    'apply_global_thresholding': lambda well_img, hyperparams: apply_global_thresholding(well_img, hyperparams),
    'apply_adaptive_thresholding': lambda well_img, hyperparams: apply_adaptive_thresholding(well_img, hyperparams['block_size'], hyperparams['offset']),
    'apply_joint_thresholding': lambda well_img, hyperparams: apply_joint_thresholding(well_img, hyperparams),
    'apply_morph_open': lambda well_img, hyperparams: apply_morph(well_img, hyperparams['kernel_size']),
    'apply_morph_close': lambda well_img, hyperparams: apply_morph(well_img, hyperparams['kernel_size'], morph_type='close'),
    'add_well_boundaries': lambda well_img, hyperparams: add_well_boundaries(well_img, hyperparams['radius'], hyperparams['thickness'], hyperparams['color']),
    'remove_artificial_well_boundary': lambda well_img, hyperparams: remove_artificial_well_boundary(well_img, hyperparams['color']),
    'mask_binary_onto_original': lambda well_img, well_saved_states: mask_binary_onto_original(well_img, well_saved_states),
    'mask_binary_inv_onto_original': lambda well_img, well_saved_states: mask_binary_inv_onto_original(well_img, well_saved_states)
}

