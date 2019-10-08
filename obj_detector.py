"""
AUTHOR: Alex Lau

SUMMARY
do all heavy-lifting jobs for object detection:
1. load in model weight
2. model inference 
3. bounding box processing
4. distance inference from inference result

REFERENCE
1. object detection + distance to object ipynb
    - https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb
2. what does cv2.dnn.blobFromImage is doing (data pre-processing step)
    - https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/

LOG
[08/10/2019]
- SSD MobileNet only support square image so far ...
- need to conversion between rectangular image to square image
"""
import os
import sys

import numpy as np
import cv2


def get_class_name(model_type = 'SSD'):
    """
    get label name for different models

    input:
        model_type -- str, 'SSD' ... etc
    """
    if model_type == 'SSD':
        label = ("background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor")
    return label


def load_model(prototxt_path, model_path):
    print('prototxt_path: {}'.format(prototxt_path))
    print('model_path: {}'.format(model_path))
    # fail fast
    assert os.path.isfile(prototxt_path), 'WRONG INPUT prototxt_path'
    assert os.path.isfile(model_path), 'WRONG INPUT model_path'
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print('MODEL LOADED IN')
    return model


def cv2_normalize(img, scale = 0.007843, mean_val = 127.53, height = 300):
    """
    handler for normalizing the image before feeding it into model 
    default is scalar mean and scalar scale for SSD MobileNet, and image input is a square

    input:
        img -- np array, (h, w, c)
        scale -- float/ list, channel-wise std
        mean_val -- float/ list, channel-wise mean
        height -- int, size of model input
    output:
        blob -- np array, (batch, c, h, w), image after pre-processing
    """
    blob = cv2.dnn.blobFromImage(img, scale, (height, height), mean_val, False)
    return blob


def feed_model(model, blob):
    model.setInput(blob, 'data')
    out = model.forward('detection_out')
    return out


def plot_prediction():
    pass

if __name__ == '__main__':
    pass

