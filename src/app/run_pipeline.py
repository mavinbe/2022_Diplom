import cv2
import mediapipe as mp
import torch

from modules.object_tracker import ObjectTracker
from modules.pose_detector import PoseDetector
from utils.general import LOGGER
from yolov5.utils.torch_utils import time_sync

PoseLandmark = mp.solutions.pose.PoseLandmark

import imageio


def calculate_newest_track_id():
    return max(object_detection_dict.keys())

def calculate_oldest_track_id():
    return min(object_detection_dict.keys())


def translate_local_to_global_coords(pose_dict, global_x, global_y):
    pose_dict = pose_dict.copy()
    for key in pose_dict:
        pose_dict[key]['x'] = int(pose_dict[key]['x'] + global_x)
        pose_dict[key]['y'] = int(pose_dict[key]['y'] + global_y)

    return pose_dict

def zoom(img, zoom_factor, center=None):
    height, width, _ = image.shape
    if center is None:
        center = (width / 2 , height / 2)
    x_left = int(center[0] - width / zoom_factor / 2)
    x_right = int(center[0] + width / zoom_factor / 2)
    y_top = int(center[1] - height / zoom_factor / 2)
    y_bottom = int(center[1] + height / zoom_factor / 2)
    img = img[y_top:y_bottom, x_left:x_right]
    try:

        img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
    except:
        print(f'[{x_left}:{x_right},{y_top}:{y_bottom}]')
        print(img.shape)

    return  img


def run(handle_image):
    global t1, success, image, t2, object_detection_dict
    img_stream = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output014.mp4")
    with PoseDetector(show_vid=False) as pose_detector:

        object_tracker = ObjectTracker(show_vid=False)

        while img_stream.isOpened():
            t1 = time_sync()

            success, image = img_stream.read()
            t2 = time_sync()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            object_detection_dict = object_tracker.inference_frame(image)
            t3 = time_sync()

            if len(object_detection_dict) > 0:
                track_id_to_track = calculate_oldest_track_id()
                detection_which_to_pose_detect = object_detection_dict[track_id_to_track]
                print(detection_which_to_pose_detect)
                cropped_image = image[detection_which_to_pose_detect[1]:detection_which_to_pose_detect[3],
                                detection_which_to_pose_detect[0]:detection_which_to_pose_detect[2]]
                pose_detect_dict = pose_detector.inference_frame(cropped_image)
                pose_detect_dict = translate_local_to_global_coords(pose_detect_dict, detection_which_to_pose_detect[0],
                                                                    detection_which_to_pose_detect[1])
                if PoseLandmark.NOSE in pose_detect_dict and PoseLandmark.LEFT_EYE in pose_detect_dict and PoseLandmark.RIGHT_EYE in pose_detect_dict:
                    print("LEFT " + str(
                        pose_detect_dict[PoseLandmark.NOSE]['z'] - pose_detect_dict[PoseLandmark.LEFT_EYE][
                            'z']) + "\t\t" + str(
                        pose_detect_dict[PoseLandmark.NOSE]['z'] - pose_detect_dict[PoseLandmark.RIGHT_EYE]['z']))
                    # cv2.circle(image, (pose_detect_dict[PoseLandmark.NOSE]['x'], pose_detect_dict[PoseLandmark.NOSE]['y']),
                    #            5,
                    #            (0, 0, 255), 2)
                    # cv2.circle(image,
                    #            (pose_detect_dict[PoseLandmark.LEFT_EYE]['x'], pose_detect_dict[PoseLandmark.LEFT_EYE]['y']),
                    #            4,
                    #            (0, 255, 0), 2)
                    # cv2.circle(image,
                    #            (pose_detect_dict[PoseLandmark.RIGHT_EYE]['x'], pose_detect_dict[PoseLandmark.RIGHT_EYE]['y']),
                    #            4,
                    #            (0, 255, 0), 2)
                    image = zoom(image, 20,
                                 (pose_detect_dict[PoseLandmark.NOSE]['x'], pose_detect_dict[PoseLandmark.NOSE]['y']))

                else:
                    print("No Nose")

                handle_image(image)
            t4 = time_sync()

            # LOGGER.info(f'DONE on hole :({(t4 - t1)*1000:.3f}ms)    read_image:({(t2 - t1)*1000:.3f}ms), object_track:({(t3 - t2)*1000:.3f}ms), pose_detect:({(t4 - t3)*1000:.3f}ms)')
    img_stream.release()


def show_image(image):
    cv2.imshow("asd", image)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        exit(0)


if __name__ == '__main__':

    run(show_image)
