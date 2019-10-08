import os
import sys
import unittest

import matplotlib.pyplot as plt
import cv2
import numpy as np

sys.path.append('..')
from img_util import load_img, reshape_img
from obj_detector import *


class ObjDetectorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup parameters
        PROTOTXT_PATH = os.path.join('..', 'model', 'MobileNetSSD_deploy.prototxt.txt')
        MODEL_PATH = os.path.join('..', 'model', 'MobileNetSSD_deploy.caffemodel')
        RGB_PATH = os.path.join('npy_test_case', 'rgb_image.npy')
        DEPTH_PATH = os.path.join('npy_test_case', 'depth_image.npy')
        # load in model
        cls.model = load_model(PROTOTXT_PATH, MODEL_PATH)
        # load in RGBD image
        cls.rgb = load_img(RGB_PATH)
        cls.d = load_img(DEPTH_PATH)
        # load class name
        cls.class_name = get_class_name(model_type = 'SSD')
        # load others
        cls.size = 300

    @classmethod
    def test_model_setup(cls):
        assert isinstance(cls.model, cv2.dnn_Net), 'WRONG MODEL TYPE (SHOULD BE cv2.dnn_Net)'

    @classmethod
    def test_cv2_normalize(cls):
        rgb = reshape_img(cls.rgb, expect_h = cls.size)
        blob_rgb = cv2_normalize(rgb)
        print('blob size: {}'.format(blob_rgb.shape))

    @classmethod
    def test_model_feed(cls):
        rgb = reshape_img(cls.rgb, expect_h = cls.size)
        blob_rgb = cv2_normalize(rgb)
        out = feed_model(cls.model, blob_rgb)
        print('model output size: {}'.format(out.shape))
        for i, arr in enumerate(out[0, 0, :, :]):
            _, cls_num, conf, xmin, ymin, xmax, ymax = arr
            print('\n####### item {} #########'.format(i+1))
            print('label index: {}'.format(cls_num))
            print('label name: {}'.format(cls.class_name[int(cls_num)]))
            print('conf: {}'.format(conf))
            print('xmin: {}'.format(xmin))
            print('ymin: {}'.format(ymin))
            print('xmax: {}'.format(xmax))
            print('ymax: {}'.format(ymax))

    @classmethod
    def test_class_name(cls):
        assert cls.class_name[0] == 'background', 'WRONG CLASS NAME'
        assert cls.class_name[1] == 'aeroplane', 'WRONG CLASS NAME'

    @classmethod
    def test_plot_prediction(cls):
        plt.hist([1,2,4,5])
        plt.show()


if __name__ == '__main__':
    unittest.main()