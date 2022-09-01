import SimpleITK as sitk
import numpy as np
import os

%matplotlib notebook
import gui

from downloaddata import fetch_data as fdata
from utilities import parameter_space_regular_grid_sampling, similarity3D_parameter_space_regular_sampling, eul2quat

OUTPUT_DIR = 'output'


def threshold_based_crop_and_bg_median(image):
    '''
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box and compute the background
    median intensity.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
        Background median intensity value.
    '''
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    bin_image = sitk.OtsuThreshold(image, inside_value, outside_value)

    # Get the median background intensity
    label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_stats_filter.SetBackgroundValue(outside_value)
    label_intensity_stats_filter.Execute(bin_image, image)
    bg_median = label_intensity_stats_filter.GetMedian(inside_value)

    # Get the bounding box of the anatomy
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(bin_image)
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return bg_median, sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box) / 2):],
                                            bounding_box[0:int(len(bounding_box) / 2)])


bg_medians, modified_data = zip(*[threshold_based_crop_and_bg_median(img) for img in data])

disp_images(modified_data, fig_size=(6, 2))