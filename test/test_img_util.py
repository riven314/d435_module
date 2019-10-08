"""
AUTHOR: Alex Lau

SUMMARY
testing functions in img_util.py

REFERENCE
1. how to import module in one level up? 
    - https://stackoverflow.com/questions/32767362/import-a-module-from-a-directory-package-one-level-up?rq=1
2. how does "import" works in python?
    - https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html

LOG
[08/10/2019]
- set up import mechanism
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.append('..')
from img_util import load_img, reshape_img

def test_reshape_img():
    # load in image
    RGB_PATH = os.path.join('npy_test_case', 'rgb_image.npy')
    DEPTH_PATH = os.path.join('npy_test_case', 'depth_image.npy')
    rgb = load_img(RGB_PATH)
    d = load_img(DEPTH_PATH)
    # reshape with expect = 300
    reshaped_rgb = reshape_img(rgb, expect_h = 300)
    reshaped_d = reshape_img(d, expect_h = 300)
    assert reshaped_rgb.shape == reshaped_d.shape == (300, 300, 3), 'WRONG IMAGE SIZE (300)'
    # reshape with expect = 200
    reshaped_rgb = reshape_img(rgb, expect_h = 200)
    reshaped_d = reshape_img(d, expect_h = 200)
    assert reshaped_rgb.shape == reshaped_d.shape == (200, 200, 3), 'WRONG IMAGE SIZE (200)'
    return reshaped_rgb, reshaped_d


if __name__ == '__main__':
    rgb, d = test_reshape_img()
    