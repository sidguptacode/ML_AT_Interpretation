import cv2
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from scipy import ndimage

from sklearn.metrics import auc

from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.filters import threshold_otsu
from skimage import img_as_float64
from scipy.signal import find_peaks
from PIL import Image, ImageEnhance, ImageOps
from PIL import ImageFilter


def perspective_correction(img, img_cords=None, ref_cords=None, box='IB1_CCBR',
                           cropped=False, plot=None):
    height, width = img.shape[:2]

    if img_cords is None:
        fname = box + '_cords.txt'
        with open(fname) as file:
            img_cords = json.load(file)

    if img_cords.get('height') is not None:
        img_cords = img_cords.copy()
        ref_cords = ref_cords.copy()

        if img_cords['height'] != height:
            x_scale = height / img_cords['height']
            for cord in img_cords:
                if 'x' in cord:
                    img_cords[cord] = int(x_scale * img_cords[cord])
                    ref_cords[cord] = int(x_scale * ref_cords[cord])
        if img_cords['width'] != width:
            y_scale = width / img_cords['width']
            for cord in img_cords:
                if 'y' in cord:
                    img_cords[cord] = int(y_scale * img_cords[cord])
                    ref_cords[cord] = int(y_scale * ref_cords[cord])

    if ref_cords is None:
        ref_cords = {'NE_x': int(width / 3.6),
                     'NE_y': int(height / 2.95),
                     'NW_x': width - int(width / 3.6),
                     'NW_y': int(height / 2.95),
                     'SE_x': int(width / 3.6),
                     'SE_y': height - int(height / 2.95),
                     'SW_x': width - int(width / 3.6),
                     'SW_y': height - int(height / 2.95)
                     }

    pts1 = np.float32([[ref_cords['NE_x'], ref_cords['NE_y']],
                       [ref_cords['NW_x'], ref_cords['NW_y']],
                       [ref_cords['SE_x'], ref_cords['SE_y']],
                       [ref_cords['SW_x'], ref_cords['SW_y']]])

    pts2 = np.float32([[img_cords['NE_x'], img_cords['NE_y']],
                       [img_cords['NW_x'], img_cords['NW_y']],
                       [img_cords['SE_x'], img_cords['SE_y']],
                       [img_cords['SW_x'], img_cords['SW_y']]])

    M = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(img, M, (width, height))

    if cropped:
        dst = dst[int(height / 3.0):height - int(height / 3.0),
              int(width / 6.0):width - int(width / 6.0)]

    if plot is None:
        return dst
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(img)
        ax1.plot((ref_cords['NE_x'], ref_cords['NW_x'], ref_cords['SE_x'], ref_cords['SW_x']),
                 (ref_cords['NE_y'], ref_cords['NW_y'], ref_cords['SE_y'], ref_cords['SW_y']),
                 marker='o', linestyle='', color='tab:green', markersize=3)

        ax1.plot((img_cords['NE_x'], img_cords['NW_x'], img_cords['SE_x'], img_cords['SW_x']),
                 (img_cords['NE_y'], img_cords['NW_y'], img_cords['SE_y'], img_cords['SW_y']),
                 marker='o', linestyle='', color='tab:red', markersize=3)
        ax1.set_title('Original Image')
        ax2.imshow(dst)
        ax2.set_title('Corrected Image')
        for ax_ in [ax1, ax2]:
            ax_.set_xticks([])
            ax_.set_yticks([])
        return dst, fig


# Image variance
# https://stackoverflow.com/questions/11456565/opencv-mean-sd-filter/11459915#11459915
def image_variance(src, ksize=3):
    img_f = img_as_float64(src, True)
    ksize = (ksize, ksize)

    img_mean = cv2.blur(img_f, ksize=ksize)
    img_sqr_mean = cv2.blur(img_f ** 2, ksize=ksize)

    var = img_sqr_mean - img_mean ** 2
    var = np.clip(var, 0, 1)
    return var


# def image_variance(img, window = 3, mode ='nearest'):
#     win_mean = ndimage.filters.uniform_filter(img, size = window, mode= mode)
#     win_sqr_mean = ndimage.filters.uniform_filter(img**2, size = window, mode= mode)
#     win_var = win_sqr_mean - win_mean**2
#     return win_var

# def contour_centroid(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _,thresh = cv2.threshold(img,127,255,0)
#     contours = cv2.findContours(thresh, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#     cnt = contours[0]
#     M = cv2.moments(cnt)
#     cx = int(M['m10']/M['m00'])
#     cy = int(M['m01']/M['m00'])
#     return (cx, cy)

def coarse_crop(img, offset=0):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray)
    bw = closing(gray > thresh, square(3))

    # minr = 0
    # minc = 0
    # maxc,maxr = img.shape[:2]
    label_img = label(bw)
    for region in regionprops(label_img, cache=False):
        if region.area > 1e5:
            minr, minc, maxr, maxc = region.bbox
            # the height cannot be larger than 400 px
            if maxr > (minr + 400):
                maxr = (minr + 400)
            break
    cimg = img[minr + offset:maxr - offset, minc + offset:maxc - offset]
    return cimg


def fine_crop(img, wlimit=0, plot=False):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.blur(gray, (300, 10))

    kernel = np.ones((10, 100), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=3)

    _, thresh = cv2.threshold(morph, 100, 255, cv2.THRESH_OTSU)

    if (cv2.__version__[0] == '4'):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = sorted(contours, key=cv2.contourArea)[-1]

    x, y, w, h = cv2.boundingRect(cnt)
    h -= 10

    if w > wlimit:
        dx = w - wlimit
        x += int(dx / 2)
        w = wlimit

    cimg = img[y:y + h, x:x + w]

    if plot:
        fig, ax = plt.subplots()
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ax.imshow(img)

        ax.set_xticks([])
        ax.set_yticks([])

        return cimg, fig
    else:
        return cimg


def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0] * (mu_g / float(nimg[0].max())), 255)
    nimg[2] = np.minimum(nimg[2] * (mu_g / float(nimg[2].max())), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0] ** 2)
    max_r = nimg[0].max()
    max_r2 = max_r ** 2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2, sum_r], [max_r2, max_r]]),
                                  np.array([sum_g, max_g]))
    nimg[0] = np.minimum((nimg[0] ** 2) * coefficient[0] + nimg[0] * coefficient[1], 255)
    sum_b = np.sum(nimg[1])
    sum_b2 = np.sum(nimg[1] ** 2)
    max_b = nimg[1].max()
    max_b2 = max_r ** 2
    coefficient = np.linalg.solve(np.array([[sum_b2, sum_b], [max_b2, max_b]]),
                                  np.array([sum_g, max_g]))
    nimg[1] = np.minimum((nimg[1] ** 2) * coefficient[0] + nimg[1] * coefficient[1], 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def sort_contours(cnts, method="left-to-right"):
    '''Returns Sorted Contours
        Available methods:
        - left-to-right (default)
        - right-to-left
        - top-to-bottom
        - bottom-to-top
        '''
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts)


def annotate_image(src, crop=None, positive=False,
                   negative=False, invert=False, plot=False,
                   add_text=True):
    '''Crop and Annotate Image'''
    if (crop is not None):
        x, y, h, w = crop
        crop_img = src[y:y + h, x:x + w]
    else:
        cimg = coarse_crop(src)
        crop_img = fine_crop(cimg, wlimit=700)

        h, w, _ = crop_img.shape

    sample_list = ['A', 'B', 'AB', 'RhD']
    if (negative):
        sample_list.append('Negative')
    if (positive):
        sample_list.append('Positive')
    if (invert):
        sample_list = list(reversed(sample_list))

    spacing = int(w / (2 * len(sample_list) + 2))
    y_pos = h - 10

    if plot:
        an_img = crop_img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.7
        font_line = 1

        if add_text:
            cv2.putText(an_img, sample_list[0], (spacing, y_pos),
                        font, font_size, (0, 0, 0), font_line, cv2.LINE_AA)
            cv2.putText(an_img, sample_list[1], (spacing * 3, y_pos),
                        font, font_size, (0, 0, 0), font_line, cv2.LINE_AA)
            cv2.putText(an_img, sample_list[2], (spacing * 6, y_pos),
                        font, font_size, (0, 0, 0), font_line, cv2.LINE_AA)
            cv2.putText(an_img, sample_list[3], (spacing * 9, y_pos),
                        font, font_size, (0, 0, 0), font_line, cv2.LINE_AA)

            if (standard):
                cv2.putText(an_img, sample_list[4], (spacing * 11, h - 10),
                            font, font_size, (0, 0, 0), font_line, cv2.LINE_AA)

        fig = plt.figure()
        ax = fig.subplots(1, 1)
        ax.imshow(an_img)

        ax.set_xticks([])
        ax.set_yticks([])

        return crop_img, sample_list, fig
    else:
        return crop_img, sample_list


def detect_droplets(src, sharpen=True, var_enchance=False, plot=False):
    '''Returns the droplet contours'''
    # Sharpen image
    img = src.copy()

    # Make code compatible with larger images
    width = img.shape[1]
    if width < 1000:
        # Perform pyramid mean shift filtering to aid the thresholding step
        if sharpen:
            shifted = cv2.pyrMeanShiftFiltering(img, 9, 51)

            for x in range(4):
                bimg = cv2.GaussianBlur(shifted, (0, 0), 3)
                shifted = cv2.addWeighted(shifted, 1.5, bimg, -0.5, 0)
        else:
            shifted = img
    else:
        scale_factor = 700 / width
        img = cv2.resize(img, None,
                         fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_AREA)
        # shifted = src.copy()
        k = 3
        # shifted = cv2.medianBlur(shifted,k)
        shifted = cv2.blur(img, (k, k))
        # shift = 51
        # shifted = cv2.addWeighted(cv2.blur(shifted,(3,shift)),0.5,
        #                           cv2.blur(shifted,(shift,3)),0.5,0)

        # for x in range(4):
        #     shifted = cv2.addWeighted(cv2.blur(shifted,(1,5-x)),0.5,
        #                               cv2.blur(shifted,(5-x,1)),0.5,0)
        # kernel = np.ones((2,2),np.uint8)
        # shifted = cv2.dilate(shifted,kernel,iterations = 1)
        # shifted = cv2.addWeighted(cv2.blur(img,(1,2)),0.5,
        #                           cv2.blur(img,(2,1)),0.5,0)

    # Convert the mean shift image to grayscale, then apply Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    if var_enchance:
        gray_v = (gray * image_variance(gray)).astype('uint8')
        thresh_v = cv2.threshold(gray_v, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.bitwise_or(thresh, thresh_v)

    # Feather white areas in larger images
    if width > 1000:
        thresh = cv2.blur(thresh, (5, 5))
        thresh[thresh < 150] = 0
        min_spot_area = 20
    else:
        min_spot_area = 50

    # find contours in the thresholded image
    if (cv2.__version__[0] == '4'):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

    contours = sort_contours(contours, method="left-to-right")

    min_y, min_x = thresh.shape
    max_y = max_x = 0
    pre_x = 0
    cnts = []
    min_area = 2500
    max_area = 7000
    boundaries = 5
    x_step = 20
    bounding_boxes = []
    cboxes = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # exclude small contours and contours at the image boundaries
        if ((w * h > min_spot_area) and
                (x > boundaries) and (y > boundaries) and
                (x + w < thresh.shape[1] - boundaries) and (y + h < thresh.shape[0] - boundaries)):
            if x <= (pre_x + x_step):
                min_x, max_x = min(x, min_x), max(x + w, max_x)
                min_y, max_y = min(y, min_y), max(y + h, max_y)
                cnts.append(contour)
            else:
                area = (max_x - min_x) * (max_y - min_y)
                if (area > min_area) and (area < max_area):
                    bounding_boxes.append((min_x, min_y, max_x, max_y))
                    cboxes.append(cnts)
                min_y, min_x = thresh.shape
                max_y = max_x = 0
                cnts = []
                min_x, max_x = min(x, min_x), max(x + w, max_x)
                min_y, max_y = min(y, min_y), max(y + h, max_y)
                cnts.append(contour)

            pre_x = x

    # Include last contour
    area = (max_x - min_x) * (max_y - min_y)
    if (area > min_area) and (area < max_area):
        bounding_boxes.append((min_x, min_y, max_x, max_y))
        cboxes.append(cnts)

    for (i, cbox) in enumerate(cboxes):
        cbox = np.concatenate(cbox, axis=0)
        cbox = cv2.convexHull(cbox)
        cboxes[i] = cbox

    contours = cboxes

    if plot:
        cbox_img = img.copy()
        for (i, box) in enumerate(bounding_boxes):
            (min_x, min_y, max_x, max_y) = box
            # area = (max_x - min_x) * (max_y - min_y)
            # print('Area of #{}: {}'.format(i+1, area))
            cv2.rectangle(cbox_img,
                          (min_x, min_y),
                          (max_x, max_y),
                          (255, 0, 0), 2)
            # cv2.putText(cbox_img, "#{}".format(i+1),
            #             (int(min_x) - 10, int(min_y)),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.8, (0, 0, 0), 2)

        # Show total contours
        c_img = img.copy()
        cv2.drawContours(c_img, contours, -1, [0, 255, 0], 2)

        # Show mask made from contours
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, contours, -1, 255, -1)
        masked = cv2.bitwise_and(img, img, mask=mask)

        fig, ax = plt.subplots(5, sharex=True)
        ax[0].imshow(shifted)
        ax[1].imshow(thresh, 'gray')
        ax[2].imshow(cbox_img)
        ax[3].imshow(c_img)
        ax[4].imshow(masked)

        for ax_ in ax:
            ax_.set_xticks([])
            ax_.set_yticks([])
        return contours, bounding_boxes, fig
    else:
        return contours, bounding_boxes


def scale_contours(contours, scale):
    """
        Shrinks or grows a contour by the given factor (float). 
        Effectively, this finds the center of the contour,
        translates the contour coordinates to that center,
        multiplies the contour coordinates by the scale,
        and re-translates the coordinates to their original position.
    """
    new_contours = []
    for contour in contours:
        moments = cv2.moments(contour)
        midX = int(round(moments["m10"] / moments["m00"]))
        midY = int(round(moments["m01"] / moments["m00"]))
        mid = np.array([midX, midY])
        contour = contour - mid
        contour = (contour * scale).astype(np.int32)
        contour = contour + mid
        new_contours.append(contour)
    return new_contours


def get_droplets(src, contours, bounding_boxes, sample_list, Offset=0,
                 wb_cor=False, masked=False, contour_shrink=1):
    img = src.copy()
    contours = scale_contours(contours, contour_shrink)

    # Make code compatible with larger images
    width = img.shape[1]
    if width > 1000:
        scale_factor = 700 / width
        img = cv2.resize(img, None,
                         fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_AREA)

    if wb_cor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = retinex_adjust(retinex(img))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if masked:
        mask = np.zeros_like(img[:, :, 0])
        cv2.drawContours(mask, contours, -1, 255, -1)
        mask = mask > 0
        mask = mask.astype(np.uint8)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        MaskedDropImages = OrderedDict()
        ExteriorMasks = OrderedDict()

    DropImages = OrderedDict()

    for (i, box) in enumerate(bounding_boxes):
        (min_x, min_y, max_x, max_y) = box

        min_x = min_x + Offset
        min_y = min_y + Offset
        max_x = max_x - Offset
        max_y = max_y - Offset

        drop = img[min_y:max_y, min_x:max_x]
        if masked:
            masked_drop = masked_img[min_y:max_y, min_x:max_x]
            exterior_mask = mask[min_y:max_y, min_x:max_x]

        try:
            DropImages[sample_list[i]] = drop
            if masked:
                MaskedDropImages[sample_list[i]] = masked_drop
                ExteriorMasks[sample_list[i]] = exterior_mask
        except:
            break

    if masked:
        return (DropImages, MaskedDropImages, ExteriorMasks)

    return DropImages


def get_droplet_variance(drop_img, percent=0.1, plot=False, **kwargs):
    drop_img_var = drop_img.copy()

    if plot:
        fig, ax = plt.subplots(2, len(drop_img.keys()), sharex=True, sharey=True,
                               gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
        enum = 0

    for sample, drop in drop_img.items():
        drop_bw = cv2.cvtColor(drop, cv2.COLOR_BGR2GRAY)

        var = image_variance(drop, **kwargs)
        var_bw = image_variance(drop_bw, **kwargs)

        if plot:
            ax[0, enum].imshow(drop)
            ax[1, enum].imshow(var_bw, 'gray')

            ax[0, enum].set_axis_off()
            ax[1, enum].set_axis_off()
            enum += 1

        h, w = var.shape[:2]
        if percent > 0:
            var = var[int(h * percent):-int(h * percent), int(w * percent):-int(w * percent)] * 100
            var_bw = var_bw[int(h * percent):-int(h * percent), int(w * percent):-int(w * percent)] * 100

        df = {'Blue': (var[..., 0].mean(), var[..., 0].std()),
              'Green': (var[..., 1].mean(), var[..., 1].std()),
              'Red': (var[..., 2].mean(), var[..., 2].std()),
              'BW': (var_bw.mean(), var_bw.std())}
        df = pd.DataFrame(df, index=['Mean', 'Std']).T
        drop_img_var[sample] = df

    df = pd.concat(drop_img_var, axis=1).T
    df.index.names = ['Test', 'Stat']
    df = df * 100
    return df


def get_histograms(images={}, mask=None,
                   smooth=True, window=10,
                   peak_prominence=7, peak_width=5,
                   plot=False):
    colors = ['Blue', 'Green', 'Red']

    histograms = {}
    index = np.linspace(0, 255, 256).astype('uint8')

    for sample in images.keys():
        image = images[sample]
        histograms[sample] = {}

        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
            histograms[sample][color] = hist.astype('uint8').flatten()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])
        histograms[sample]['BW'] = hist.astype('uint8').flatten()

        histograms[sample] = pd.DataFrame(histograms[sample], index=index)
        # exclude masked pixels
        histograms[sample].iloc[0] = 0

    histograms = pd.concat(histograms, axis=0, names=['Sample', 'Bin'])

    if smooth:
        histograms = histograms.rolling(window=window, center=True).mean()
        histograms = histograms.fillna(0)

    grp = histograms.groupby(by='Sample')
    scores = {}
    pks = {}
    for g, dg in grp:
        peaks = dg.apply(lambda x: find_peaks(x,
                                              prominence=peak_prominence,
                                              width=peak_width)).apply(pd.Series)
        # peaks = peaks.loc[:,0].apply(pd.Series).fillna(255)
        peaks = peaks.loc[0].apply(pd.Series).fillna(255)
        if len(peaks.columns) == 1:
            peaks[1] = 255
        _peaks = peaks.mean(axis=1)

        scores[g] = {}
        for col in histograms.columns:
            scores[g][col] = dg[col].iloc[int(_peaks[col]):].sum() / dg[col].sum()
    scores = pd.DataFrame(scores) * 100

    if plot:
        ax = histograms.plot(color=['tab:blue', 'tab:green',
                                    'tab:red', 'black'])

        peaks = histograms.apply(lambda x: find_peaks(x,
                                                      prominence=peak_prominence,
                                                      width=peak_width)).apply(pd.Series)
        peaks = peaks.loc[:, 0].apply(pd.Series).T

        ys = {}
        cols = peaks.columns
        for col in peaks.columns:
            index = peaks[col].dropna().to_numpy().astype('uint16')
            ys[col] = histograms[col].iloc[index].reset_index(drop=True)
        ys = pd.concat(ys, axis=1)
        peaks = peaks.merge(ys, how='left',
                            left_index=True, right_index=True)

        colors = [col.lower() for col in cols]
        colors = colors[:-1] + ['gray']
        for i, col in enumerate(cols):
            peaks.plot(x=col + '_x', y=col + '_y',
                       marker='x', linestyle='',
                       legend=False, color=colors[i], ax=ax)

        ax.set_xlabel('Sample')
        ticks = (np.linspace(0, len(histograms), 2 * int(len(histograms) / 256) + 1)).astype('uint32')
        t = ticks % 255
        t[t > 10] = 0
        ticks = (ticks - t)
        ticks_ = ticks.astype('<U11')
        ticks_[(ticks / 128) % 2 == 1] = histograms.index.get_level_values(0).unique().to_numpy()
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks_)
    scores = scores.T
    scores.index.names = ['Test']
    return histograms, scores


def get_pixels(sample_list=[], images={}, masked=False):
    '''Measure each pixel, returns a pandas dataframe'''
    output = ['Blue', 'Green', 'Red', 'BW']

    df_pix = {}
    for sample in sample_list:
        try:
            image = images[sample]
        except KeyError:
            print('Missing image for sample {}'.format(sample))

        d_d = {}
        for i in range(3):
            d_d[output[i]] = image[..., i].flatten()

        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        d_d[output[-1]] = image_bw.flatten()
        _df = pd.DataFrame(d_d, columns=output)

        if masked:
            _df = _df.loc[_df['BW'] > 0]

        df_pix[sample] = _df.reset_index(drop=True)

    df_pix = pd.concat(df_pix,
                       axis=1,
                       sort=False,
                       keys=sample_list).apply(pd.to_numeric, downcast='float')

    return df_pix


def measure_pixels(df_pixel, sample_list, **kwargs):
    '''Returns pandas dataframes'''
    # output = ['Row', 'Red', 'Green','Blue', 'BW' ]
    df_pix = df_pixel.copy()
    df_mean = df_pix.mean().unstack(level=[1]).reindex(index=sample_list)  # [output[1:]]

    df_std = df_pix.std().unstack(level=[1]).reindex(index=sample_list)  # [output[1:]]

    df_range = (df_pix.max() - df_pix.min()).unstack(level=[1]).reindex(index=sample_list)  # [output[1:]]

    if 'lower' in kwargs:
        ql = kwargs['lower']
    else:
        ql = 0.25
    if 'mid' in kwargs:
        qm = kwargs['mid']
    else:
        qm = 0.5
    if 'higher' in kwargs:
        qh = kwargs['higher']
    else:
        qh = 0.75

    if 'plot' in kwargs:
        plot = kwargs['plot']
    else:
        plot = False
    if 'normalize' in kwargs:
        normalize = kwargs['normalize']
    else:
        normalize = True
    if 'strech' in kwargs:
        df_pix = 255 * ((df_pix - df_pix.min()) / (df_pix.max() - df_pix.min()))

    df_l = df_pix.quantile(q=ql)
    df_h = df_pix.quantile(q=qh)

    if qm:
        df_m = df_pix.quantile(q=qm)
        df_D = ((df_h - df_l) / (2.0 * df_m))
    else:
        df_D = (df_h - df_l) / 255

    df_D = df_D.unstack(level=[1]).reindex(index=sample_list)  # [output[1:]]

    if normalize:
        df_mean = df_mean / 255.0
        df_std = df_std / 255.0
        df_range = df_range / 255.0
        ylim = (-0.05, 1.05)
        ylabel = 'Normalized Channel'
    else:
        ylim = (-0.05, 255.05)
        ylabel = 'Channel'

    df_mean.index.names = ['Test']
    df_std.index.names = ['Test']
    df_range.index.names = ['Test']
    df_D.index.names = ['Test']

    if plot:
        colors = ['r', 'g', 'b', 'black']

        ax1 = df_mean.plot(kind='bar', alpha=0.75, rot=0,
                           color=colors, yerr=df_std,
                           title='Average', ylim=ylim)
        ax1.set_ylabel(ylabel)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.87, 1])

        # df_std.plot(kind='bar',alpha=0.75, rot=0, color = colors, title='Std' )

        ax2 = df_range.plot(kind='bar', alpha=0.75, rot=0,
                            color=colors, title='Range',
                            ylim=ylim)
        ax2.set_ylabel(ylabel)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.87, 1])

        ax3 = df_D.plot(kind='bar', alpha=0.75, rot=0,
                        color=colors,
                        title='({:.0f}%-{:.0f}%)/(2*{:.0f}%)'.format(qh * 100, ql * 100, qm * 100))
        ax3.set_ylabel('({:.0f}%-{:.0f}%)/(2*{:.0f}%)'.format(qh * 100, ql * 100, qm * 100))
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.87, 1])

        return df_mean, df_std, df_range, df_D, fig
    else:
        return df_mean, df_std, df_range, df_D


def get_report_quant(df_D, channel='Green', cut_off=0.152):
    '''Returns a pandas dataframe of the final report
        Available channels:
        - Red
        - Green
        - Blue
        - BW (default)
        '''
    truth_table = np.array([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="bool")

    truth_table = pd.DataFrame(truth_table,
                               index=["A", "B", "AB", "RhD", "Negative", "Positive"],
                               columns=["A-", "A+", "B-", "B+",
                                        "AB-", "AB+", "O-", "O+",
                                        "Aw/Bw-", "Aw/Bw+"])

    truth_table = truth_table.loc[df_D.index]

    report = (df_D > cut_off).loc[truth_table.index.to_numpy()]
    res = report.copy().T
    res['Phenotype'] = np.nan
    # res = res.T
    for col in report.columns:
        for col_ in truth_table.columns:
            if report[col].equals(truth_table[col_]):
                res.loc[col, 'Phenotype'] = col_
                break
    report = pd.concat([df_D, res.T.loc[['Phenotype']]], axis=0)
    result = report.loc['Phenotype', channel]
    return report, result


def get_report_hist(hist_scores, channel='Red', cut_off=10):
    '''Returns a pandas dataframe of the final report
        Available channels:
        - Red (default)
        - Green
        - Blue
        - BW
        '''
    truth_table = np.array([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="bool")

    truth_table = pd.DataFrame(truth_table,
                               index=["A", "B", "AB", "RhD", "Negative", "Positive"],
                               columns=["A-", "A+", "B-", "B+",
                                        "AB-", "AB+", "O-", "O+",
                                        "Aw/Bw-", "Aw/Bw+"])

    truth_table = truth_table.loc[hist_scores.index]

    report = (hist_scores > cut_off).loc[truth_table.index.to_numpy()]
    res = report.copy().T
    res['Phenotype'] = np.nan
    # res = res.T
    for col in report.columns:
        for col_ in truth_table.columns:
            if report[col].equals(truth_table[col_]):
                res.loc[col, 'Phenotype'] = col_
                break
    report = pd.concat([hist_scores, res.T.loc[['Phenotype']]], axis=0)
    result = report.loc['Phenotype', channel]
    return report, result


def get_report_variance(df_var, stat='Mean',
                        channel='Red', cut_off=1.5365):
    '''Returns a pandas dataframe of the final report
        Available channels:
        - Red (default)
        - Green
        - Blue
        - BW
        '''
    df_var = df_var.groupby(by='Stat').get_group(stat).reset_index(drop=True, level=1)
    truth_table = np.array([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="bool")

    truth_table = pd.DataFrame(truth_table,
                               index=["A", "B", "AB", "RhD", "Negative", "Positive"],
                               columns=["A-", "A+", "B-", "B+",
                                        "AB-", "AB+", "O-", "O+",
                                        "Aw/Bw-", "Aw/Bw+"])

    truth_table = truth_table.loc[df_var.index]

    report = (df_var > cut_off).loc[truth_table.index.to_numpy()]
    res = report.copy().T
    res['Phenotype'] = np.nan
    # res = res.T
    for col in report.columns:
        for col_ in truth_table.columns:
            if report[col].equals(truth_table[col_]):
                res.loc[col, 'Phenotype'] = col_
                break

    report = pd.concat([df_var, res.T.loc[['Phenotype']]], axis=0)
    result = report.loc['Phenotype', channel]
    return report, result


def get_hematocrit(src,
                   low_border=0.15, high_border=0.85,
                   calibration_values=None):
    img = src.copy()
    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    df_pix = pd.Series(image_bw.flatten())
    df_pix = df_pix.sort_values().reset_index(drop=True)
    df_pix = 255 * ((df_pix - df_pix.min()) / (df_pix.max() - df_pix.min()))
    df_pix = df_pix.to_frame().dropna().reset_index()
    df_pix.columns = ['Norm Index', 'Norm Sorted Pixels']
    df_pix['Norm Index'] = df_pix['Norm Index'] / df_pix['Norm Index'].max()

    yy = df_pix.loc[((df_pix['Norm Index'] > low_border) &
                     (df_pix['Norm Index'] < high_border)),
                    'Norm Sorted Pixels'].to_numpy()
    xx = df_pix.loc[((df_pix['Norm Index'] > low_border) &
                     (df_pix['Norm Index'] < high_border)),
                    'Norm Index'].to_numpy()
    auc_ = auc(xx, yy)

    if calibration_values is not None:
        a, b = calibration_values
        hematocrit = (auc_ - a) / b
        return auc_, hematocrit * 2 / 3.3  # Change after calibration
    else:
        return auc_


def get_report_image(images={}, result=None,
                     image_name=None, plot=False, fscale=1):
    temp_list = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7 * fscale
    font_line = 1 * fscale
    bordersize = 5 * fscale

    h = 100 * fscale
    w = 50 * fscale

    sample_list = list(images.keys())

    if len(sample_list) == 4:
        sample_list = ['A', 'B', 'AB', 'RhD']
    elif len(sample_list) == 5 and 'Merquat' in sample_list:
        sample_list = ['A', 'B', 'AB', 'RhD', 'Merquat']
    elif len(sample_list) == 5 and 'Negative' in sample_list:
        sample_list = ['A', 'B', 'AB', 'RhD', 'Negative']

    for sample in sample_list:
        temp_im = cv2.resize(images[sample], (w, h))

        temp_im = cv2.copyMakeBorder(temp_im, borderType=cv2.BORDER_CONSTANT,
                                     top=bordersize * (len(sample_list) + 1), bottom=bordersize,
                                     left=bordersize, right=bordersize, value=[255, 255, 255])

        # get boundary of this text
        if sample == 'Merquat':
            auc_, hematocrit = get_hematocrit(images[sample],
                                              calibration_values=(155.421, -0.741))
            sample = '+'

        textsize = cv2.getTextSize(sample, font, font_size, 2)[0]
        # get coords based on boundary
        textX = int((temp_im.shape[1] - textsize[0]) / 2)
        textY = int((temp_im.shape[0] + textsize[1]) / 2)

        cv2.putText(temp_im, sample, (textX, bordersize * len(sample_list)),
                    font, font_size, (0, 0, 0), font_line, cv2.LINE_AA)
        temp_list.append(temp_im)

    _h, _w, _c = temp_im.shape

    if (result is not None):
        blank_image = np.zeros((_h, _w, _c), np.uint8) + 255

        cv2.putText(blank_image, 'Blood', (5 + 2 * (fscale - 1), bordersize * 4), font,
                    font_size - 0.1 * fscale, (0, 0, 0), font_line, cv2.LINE_AA)
        cv2.putText(blank_image, 'Type:', (5 + 2 * (fscale - 1), int(_h / 3)), font,
                    font_size - 0.1 * fscale, (0, 0, 0), font_line, cv2.LINE_AA)

        if not isinstance(result, str):
            result = 'NaN'

        textsize = cv2.getTextSize(result, font, font_size, 2)[0]
        textX = int((int(temp_im.shape[1]) - int(textsize[0])) / 2)
        textY = int((int(temp_im.shape[0]) + int(textsize[1])) / 2)

        cv2.putText(blank_image, result, (textX, int(_h / 1.6)),
                    font, font_size, (0, 0, 0), font_line, cv2.LINE_AA)

        try:
            hem_val = '{:0.1f}%'.format(hematocrit)
            _h_1 = int(_h / 1.25)
            _h_2 = int(_h / 1.07)
            _h_3 = int(_h / 1.5)

            cv2.putText(blank_image, 'Hematocrit:',
                        (5, _h_1), font, font_size / 2.15, (0, 0, 0),
                        font_line, cv2.LINE_AA)
            cv2.putText(blank_image, hem_val,
                        (textX - 20, _h_2), font, font_size / 1.5, (0, 0, 0),
                        font_line, cv2.LINE_AA)
            cv2.line(blank_image, (0, _h_3), (_w, _h_3), (0, 0, 0), 3 + 2 * (fscale - 1))
        except UnboundLocalError:
            pass

        cv2.line(blank_image, (0, 0), (0, _h), (0, 0, 0), 3 + 2 * (fscale - 1))

        temp_list.append(blank_image)

    image = np.concatenate(temp_list, axis=1)
    if plot:
        if (image_name != ''):
            fig = plt.figure(image_name)
        else:
            fig = plt.figure()
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

        fig.canvas.draw()

    return image


def get_indiv_images(images={}, exterior_masks={}, result=None,
                     image_name=None, plot=False, fscale=1, Offset=0):
    temp_list = []
    exterior_masks_list = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7 * fscale
    font_line = 1 * fscale
    bordersize = 5 * fscale

    h = 100 * fscale + 2*Offset
    w = 50 * fscale + 2*Offset

    sample_list = list(images.keys())

    if len(sample_list) == 4:
        sample_list = ['A', 'B', 'AB', 'RhD']
    elif len(sample_list) == 5 and 'Merquat' in sample_list:
        sample_list = ['A', 'B', 'AB', 'RhD', 'Merquat']
    elif len(sample_list) == 5 and 'Negative' in sample_list:
        sample_list = ['A', 'B', 'AB', 'RhD', 'Negative']

    for sample in sample_list:
        temp_im = images[sample]
        exterior_mask = exterior_masks[sample]
        
        # We are given a masked image. In this step, we will take the maximum pixel value
        # inside the blood droplet, and make that the mask's background. This is
        # to assist the Otsu binarization that happens when we create the dataset.
        well_img = cv2.cvtColor(temp_im, cv2.COLOR_BGR2GRAY)
        well_img = np.array(well_img)
        max_pixel_val = np.max(well_img)
        if max_pixel_val < 150:
            max_pixel_val = 150
        well_img[well_img==0] = max_pixel_val

        well_img = cv2.resize(well_img, (w, h))
        exterior_mask_resized = cv2.resize(exterior_mask, (w, h))
        exterior_mask_resized = (exterior_mask_resized > 0.5) * 1
        temp_list.append(well_img)
        exterior_masks_list.append(exterior_mask_resized)

    _h, _w = well_img.shape

    return temp_list, exterior_masks_list, sample_list

