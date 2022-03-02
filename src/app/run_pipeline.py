import cv2
import torch

from modules.object_tracker import ObjectTracker
from dataset.yolo5_dataset import LoadImages
from yolov5.utils.torch_utils import time_sync



with torch.no_grad():
    img_stream = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output015.mp4")
    object_tracker = ObjectTracker()

    while img_stream.isOpened():
        t1 = time_sync()
        success, im0 = img_stream.read()
        t2 = time_sync()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        t3 = time_sync()
        detection_dict = object_tracker.inference_frame(im0)
        t4 = time_sync()
        print(detection_dict)
        # LOGGER.info(f'DONE on prepare:({(t4 - t1)*1000:.3f}ms)    read:({(t2 - t1)*1000:.3f}ms), inference:({(t4 - t3)*1000:.3f}ms)')

img_stream.release()