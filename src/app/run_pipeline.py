import cv2
import mediapipe as mp
import torch

from modules.object_tracker import ObjectTracker
from dataset.yolo5_dataset import LoadImages
from modules.pose_detector import PoseDetector
from yolov5.utils.torch_utils import time_sync

PoseLandmark = mp.solutions.pose.PoseLandmark

img_stream = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output015.mp4")
with PoseDetector() as pose_detector:

    object_tracker = ObjectTracker()

    while img_stream.isOpened():
        t1 = time_sync()
        success, image = img_stream.read()
        t2 = time_sync()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        t3 = time_sync()
        detection_dict = object_tracker.inference_frame(image)
        t4 = time_sync()
        #print(detection_dict)
        # LOGGER.info(f'DONE on prepare:({(t4 - t1)*1000:.3f}ms)    read:({(t2 - t1)*1000:.3f}ms), inference:({(t4 - t3)*1000:.3f}ms)')
        for track_id, detection in detection_dict.items():
            pose_dict = pose_detector.inference_frame(image[detection[1]:detection[3], detection[0]:detection[2]])
            for key in pose_dict:
                pose_dict[key]['x'] = int(pose_dict[key]['x'] + detection[0])
                pose_dict[key]['y'] = int( pose_dict[key]['y'] + detection[1])



            print(pose_dict[PoseLandmark.NOSE])
            break

img_stream.release()