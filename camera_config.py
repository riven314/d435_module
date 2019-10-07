"""
AUTHOR: Alex Lau

SUMMARY
Configure the RGBD camera handler before streaming, parameters include:
1. depth scale for depth accuracy
2. resolution
3. color format
4. FPS
5. minimum depth distance
6. postprocessing on depth
7. realtime streaming for sample tests

LOG
[06/10/2019]
- current RGB must have same resolution as D
- parameters FPS has no impact on actual FPS
- adjust depth accuracy
- check if GPU is connected
"""
import os
import sys
import time

import pyrealsense2 as rs
import numpy as np
import cv2

from img_stream import stream_camera

class RGBDhandler:
    def __init__(self, rgb_res, rgb_format, depth_res, depth_format, fps):
        """
        input:
            rgb_res, depth_res - tup, (width, height) e.g. (320, 240), (640, 480), (848, 480), (1280, 720)
            rgb_format, depth_format -- str, e.g. bgr8, z16 ... etc.
            fps -- int, frames per second

        e.g. 
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        doc of format you can take:
        https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.format.html#pyrealsense2.format.rgb8
        """
        self.RGB_RESOLUTION = rgb_res
        self.RGB_FORMAT = rgb_format
        self.DEPTH_RESOLUTION = depth_res
        self.DEPTH_FORMAT = depth_format
        self.FPS = fps
        self.config = self._setup_config()
        self.depth_scale = self.get_depth_scale()

    def _setup_config(self):
        config = rs.config()
        rgb_w, rgb_h = self.RGB_RESOLUTION
        depth_w, depth_h = self.DEPTH_RESOLUTION
        rgb_format = self._setup_format(self.RGB_FORMAT)
        depth_format = self._setup_format(self.DEPTH_FORMAT)
        fps = self.FPS
        config.enable_stream(rs.stream.depth, depth_w, depth_h, depth_format, fps)
        config.enable_stream(rs.stream.color, rgb_w, rgb_h, rgb_format, fps)
        print('Configuration Setup Completed!')
        return config

    def _setup_format(self, format_str):
        assert format_str in ['bgr8', 'z16'], 'WRONG FORMAT INPUT: {}'.format(format_str)
        if format_str == 'bgr8':
            return rs.format.bgr8
        elif format_str == 'z16':
            return rs.format.z16
        else:
            print('IT SHOULDNT HAPPEN...')
            return None

    def get_depth_scale(self):
        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        pipeline.stop()
        return depth_scale

    def get_config_info(self):
        print('\n########## RGB ##########')
        print('(w x h): {}'.format(self.RGB_RESOLUTION))
        print('format: {}'.format(self.RGB_FORMAT))
        print('\n########## DEPTH ##########')
        print('(w x h): {}'.format(self.DEPTH_RESOLUTION))
        print('format: {}'.format(self.DEPTH_FORMAT))
        print('scale: {}'.format(self.depth_scale))
        print('\n########## OTHERS ##########')
        print('fps: {}'.format(self.FPS))

    def test_streamline(self, frame_limit, is_process_depth = False):
        """
        streamline until # frame = frame_limit, apply depth postprocessing if is_process_depth = True
        """
        stream_camera(config = self.config, frame_limit = frame_limit, is_process_depth = is_process_depth)

    def get_snapshot_np(self):
        """
        take a snapshot from streamline (after warmup), and then output the snapshot (as numpy array)

        output:
            color_image -- np array, (height, width, channel) (uint 8)
            depth_image -- np array, (height, width, channel) (uint 8)
        """
        color_image, depth_image = stream_camera(config = self.config, frame_limit = 1, is_process_depth= False)
        return color_image, depth_image


if __name__ == '__main__':
    resolution = (1280, 720)
    rs_handler = RGBDhandler(resolution, 'bgr8', resolution, 'z16', 30)
    rs_handler.get_config_info()
    rs_handler.test_streamline(frame_limit = 200, is_process_depth = False)
    #color_image, depth_image = rs_handler.get_snapshot_np()