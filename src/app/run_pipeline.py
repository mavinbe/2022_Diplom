from os import wait

import cv2
import mediapipe as mp
import numpy as np
import torch

from modules.object_tracker import ObjectTracker
from modules.pose_detector import PoseDetector
from utils.general import LOGGER

from modules.pysical_models.new_position_max_speed_constrained import NewPositionMaxSpeedConstrained
from yolov5.utils.torch_utils import time_sync

PoseLandmark = mp.solutions.pose.PoseLandmark

import imageio


def calculate_newest_track_id(object_detection_dict):
    return max(object_detection_dict.keys())


def calculate_oldest_track_id(object_detection_dict):
    return min(object_detection_dict.keys())


def translate_local_to_global_coords(pose_dict, global_x, global_y):
    pose_dict = pose_dict.copy()
    for key in pose_dict:
        pose_dict[key]['x'] = int(pose_dict[key]['x'] + global_x)
        pose_dict[key]['y'] = int(pose_dict[key]['y'] + global_y)

    return pose_dict


def zoom(img, target_box):
    image_original_shape = img.shape
    height, width, _ = img.shape
    img = img[target_box[0]:target_box[1], target_box[2]:target_box[3]]
    #print(image_original_shape)
    #print(target_box)
    try:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    except:

        print(img.shape)

    return img


def determ_position_by_landmark_from_pose_detection(pose_detect_dict, landmark):
    if PoseLandmark.NOSE in pose_detect_dict:
        return pose_detect_dict[PoseLandmark.NOSE]['x'], pose_detect_dict[PoseLandmark.NOSE]['y']


def static_zoom_target_box(image_shape, zoom_factor, center):
    height, width, _ = image_shape
    if center is None:
        center = (width / 2, height / 2)
    y_top = int(center[1] - height / zoom_factor / 2)
    y_bottom = int(center[1] + height / zoom_factor / 2)
    x_left = int(center[0] - width / zoom_factor / 2)
    x_right = int(center[0] + width / zoom_factor / 2)

    # don't go out of image
    if y_top < 0:
        y_bottom += 0 - y_top
        y_top = 0
    if x_left < 0:
        x_right += 0 - x_left
        x_left = 0
    if y_bottom > height:
        y_top += height - y_bottom
        y_bottom = height
    if x_right > width:
        x_left += width - x_right
        x_right = width
    return y_top, y_bottom, x_left, x_right


def calculate_face_direction(pose_detect_dict):
    ## Sample code
    # if PoseLandmark.NOSE in pose_detect_dict and PoseLandmark.LEFT_EYE in pose_detect_dict and PoseLandmark.RIGHT_EYE in pose_detect_dict:
    #     print("LEFT " + str(
    #         pose_detect_dict[PoseLandmark.NOSE]['z'] - pose_detect_dict[PoseLandmark.LEFT_EYE][
    #             'z']) + "\t\t" + str(
    #         pose_detect_dict[PoseLandmark.NOSE]['z'] - pose_detect_dict[PoseLandmark.RIGHT_EYE]['z']))
    #     # cv2.circle(image, (pose_detect_dict[PoseLandmark.NOSE]['x'], pose_detect_dict[PoseLandmark.NOSE]['y']),
    #     #            5,
    #     #            (0, 0, 255), 2)
    #     image = zoom(image, 20,
    #                  (pose_detect_dict[PoseLandmark.NOSE]['x'], pose_detect_dict[PoseLandmark.NOSE]['y']))
    pass


def run(handle_image):
    
    img_stream = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output014.mp4")
    #img_stream.set(cv2.CAP_PROP_POS_FRAMES, 290)
    with PoseDetector(show_vid=False) as pose_detector:

        object_tracker = ObjectTracker(show_vid=False)
        frame_count = 0
        last_target_box = None
        position_model = None
        while img_stream.isOpened():
            if frame_count > 1600:
                exit()
            t1 = time_sync()

            success, image = img_stream.read()
            height, width, _ = image.shape
            #image = cv2.flip(image, 1)
            t2 = time_sync()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            if position_model is None:

                position_model = NewPositionMaxSpeedConstrained(time_sync(), np.asarray((int(width/2), int(height/2))), 20)
            frame_count += 1
            object_detection_dict = object_tracker.inference_frame(image)
            t3 = time_sync()

            if len(object_detection_dict) > 0:
                track_id_to_track = calculate_newest_track_id(object_detection_dict)
                detection_which_to_pose_detect = object_detection_dict[track_id_to_track]
                # print(detection_which_to_pose_detect)
                cropped_image = image[detection_which_to_pose_detect[1]:detection_which_to_pose_detect[3],
                                detection_which_to_pose_detect[0]:detection_which_to_pose_detect[2]]
                pose_detect_dict = pose_detector.inference_frame(cropped_image)
                pose_detect_dict_in_global = translate_local_to_global_coords(pose_detect_dict, detection_which_to_pose_detect[0],
                                                                    detection_which_to_pose_detect[1])
                #if 9 in pose_detect_dict_in_global and 10 in pose_detect_dict_in_global:
                    #print([pose_detect_dict_in_global[9], pose_detect_dict_in_global[10]])
                target_position = determ_position_by_landmark_from_pose_detection(pose_detect_dict_in_global, PoseLandmark.NOSE)
                #print(target_position)
                if target_position:
                    position_model.move_to_target(target_position, time_sync())
                    #print(position_model.get_position())
                else:
                    print("No Landmark")
                target_box = static_zoom_target_box(image.shape, 20, position_model.get_position())
                #print(f' #################   {(int(position_model.get_position()[0]), int(position_model.get_position()[1]))}')
                # cv2.circle(image, (int(position_model.get_position()[0]), int(position_model.get_position()[1])),
                #                           3,
                #                           (0, 0, 255), 5)
                # cv2.circle(image, target_position,
                #            7,
                #            (255, 0, 0), 2)
                image = zoom(image, target_box)

                handle_image(image)
            t4 = time_sync()

            LOGGER.info(
                f'frame_count {frame_count} DONE on hole :({(t4 - t1) * 1000:.3f}ms)    read_image:({(t2 - t1) * 1000:.3f}ms), object_track:({(t3 - t2) * 1000:.3f}ms), pose_detect:({(t4 - t3) * 1000:.3f}ms)')
    img_stream.release()


def show_image(image):
    cv2.imshow("asd", image)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        exit(0)


if __name__ == '__main__':
    run(show_image)
