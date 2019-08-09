# imports
import os
import sys

import numpy as np

#try:
from timagetk.util import data_path
from timagetk.components import imread, imsave
from timagetk.plugins import linear_filtering, morphology
from timagetk.plugins import h_transform
from timagetk.plugins import region_labeling, segmentation
from timagetk.plugins import labels_post_processing
#except ImportError:
#    raise ImportError('Import Error')

#out_path = './results/' # to save results
#if not os.path.isdir(out_path):
#    new_fold = os.path.join(os.getcwd(),'results')
#    os.mkdir(new_fold)

# we consider an input image
# SpatialImage instance
input_img = imread(sys.argv[1])

print input_img

if input_img.dtype==np.uint8:
    input_img = input_img.astype(np.uint16)
    input_img *= 256

# optional denoising block
smooth_img = linear_filtering(input_img, std_dev=float(sys.argv[4]),
                              method='gaussian_smoothing')

asf_img = morphology(smooth_img, max_radius=1,
                     method='co_alternate_sequential_filter')

#opening = morphology(input_img, radius=10,
#                     method='opening')

#closing = morphology(input_img, radius=10,
#                     method='dilation')

#closing2 = morphology(input_img, radius=1,
#                     method='dilation')
"""
closing = grey_opening(input_img>30, 3)
closing2 = grey_opening(input_img>30, 10)
opening = grey_erosion(input_img>30, 3)
opening2 = grey_erosion(input_img>30, 10)

"""
threshold = int(sys.argv[3])

ext_img = h_transform(asf_img, h=threshold,
                      method='h_transform_min')

print 'done transform'

con_img = region_labeling(ext_img, low_threshold=1,
                          high_threshold=threshold,
                          method='connected_components')

print 'done labelling'

seg_img = segmentation(smooth_img, con_img, control='first',
                       method='seeded_watershed')

print 'done seg'

# optional post processig block
#pp_img = labels_post_processing(seg_img, radius=1,
#                                iterations=1,
#                                method='labels_erosion')

res_name = 'example_segmentation.tif'
imsave(sys.argv[2], seg_img)
