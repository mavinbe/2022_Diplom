import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

from utils.general import check_imshow
from yolov5.utils.torch_utils import time_sync

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class PoseDetector:
    def __init__(self, show_vid=False):
        # Check if environment supports image displays
        if show_vid:
            show_vid = check_imshow()
        self.show_vid = show_vid

        self.pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def __enter__(self):
        self.pose.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pose.__exit__()

    def inference_frame(self, image):
        start = time.perf_counter()

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        landmark_list = results.pose_landmarks
        result_dict = {}
        height, width, _ = image.shape
        #print(image.shape)
        if type(landmark_list) is landmark_pb2.NormalizedLandmarkList:
            for idx, landmark in enumerate(landmark_list.landmark):
              result_dict[idx] = {'x': landmark.x * width, 'y': landmark.y * height, 'z': landmark.z, 'visibility': landmark.visibility}
              #print(type(landmark))


        if self.show_vid:
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                exit()

        #inference_time = time.perf_counter() - start
        #print('%.2f ms' % (inference_time * 1000))

        return result_dict


if __name__ == '__main__':
    cap = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output015.mp4")
    with PoseDetector() as pose_detector:

        print("sdf")

        while cap.isOpened():
            t1 = time_sync()
            success, im0 = cap.read()
            t2 = time_sync()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            t3 = time_sync()
            result_list = pose_detector.inference_frame(im0)
            t4 = time_sync()
            print(result_list)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                break
            # LOGGER.info(f'DONE on prepare:({(t4 - t1)*1000:.3f}ms)    read:({(t2 - t1)*1000:.3f}ms), prepare:({(t3 - t2)*1000:.3f}ms), inference:({(t4 - t3)*1000:.3f}ms)')

    cap.release()
