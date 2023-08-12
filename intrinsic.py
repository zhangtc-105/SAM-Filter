import cv2
import numpy as np 
import pyrealsense2 as rs
import open3d as o3d
from PIL import Image
import time

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color,640,480, rs.format.bgr8, 30)
align_to = rs.stream.color
align = rs.align(align_to)
pipeline.start(config)
try:
    while True:
        frames= pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        print(depth_image.shape)
        print(color_image.shape)
        
        np.save("depth.npy",depth_image)
        cv2.imwrite("Images/color.png",color_image)
        time.sleep(1)
finally:
    pipeline.stop()