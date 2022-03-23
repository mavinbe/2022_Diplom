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


def get_track_id(object_detection_dict, track_id_shift):
    return max(object_detection_dict.keys())

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


def snapshot(last_image, pose_detect_dict):
    file_name = './data/'+str(time_sync())+'.jpg'
    print(file_name)
    print(pose_detect_dict)
    cv2.imwrite(file_name, last_image)




def run():
    global t1, success, image, t2
    img_stream = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output014.mp4")

    with PoseDetector(show_vid=True) as pose_detector:

        object_tracker = ObjectTracker(show_vid=True)
        frame_count = 0

        position_model = None
        track_id_shift = 0
        frames_shift = 0

        step = True
        while img_stream.isOpened():

            if step:
                step = False
                success, image = img_stream.read()
                height, width, _ = image.shape
                #image = cv2.flip(image, 1)


                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                if position_model is None:

                    position_model = NewPositionMaxSpeedConstrained(time_sync(), np.asarray((int(width/2), int(height/2))), 20)
                frame_count += 1
                object_detection_dict = object_tracker.inference_frame(image)


                if len(object_detection_dict) > 0:
                    track_id_to_track = calculate_newest_track_id(object_detection_dict)
                    detection_which_to_pose_detect = object_detection_dict[track_id_to_track]
                    # print(detection_which_to_pose_detect)
                    cropped_image = image[detection_which_to_pose_detect[1]:detection_which_to_pose_detect[3],
                                    detection_which_to_pose_detect[0]:detection_which_to_pose_detect[2]]
                    pose_detect_dict = pose_detector.inference_frame(cropped_image)
                    pose_detect_dict_in_global = translate_local_to_global_coords(pose_detect_dict, detection_which_to_pose_detect[0],
                                                                        detection_which_to_pose_detect[1])
                    # if 9 in pose_detect_dict and 10 in pose_detect_dict:
                    #     print([pose_detect_dict[9], pose_detect_dict[10]])
                    target_position = determ_position_by_landmark_from_pose_detection(pose_detect_dict_in_global, PoseLandmark.NOSE)


                    #print(f' #################   {(int(position_model.get_position()[0]), int(position_model.get_position()[1]))}')
                    # cv2.circle(image, (int(position_model.get_position()[0]), int(position_model.get_position()[1])),
                    #                           3,
                    #                           (0, 0, 255), 5)
                    # cv2.circle(image, target_position,
                    #            7,
                    #            (255, 0, 0), 2)


                    #cv2.imshow("asd", image)
            key = cv2.waitKey(10)
            if key == ord('q'):  # q to quit
                print(f'key {key}')
                exit(0)
            elif key == 82:
                track_id_shift = 1
            elif key == 84:
                track_id_shift = -1
            elif key == ord('g'):
                frames_shift = -1
            elif key == ord('f'):
                frames_shift = -5
            elif key == ord('d'):
                frames_shift = -20
            elif key == ord('s'):
                frames_shift = -100
            elif key == ord('h'):
                frames_shift = 1
            elif key == ord('j'):
                frames_shift = 5
            elif key == ord('k'):
                frames_shift = 20
            elif key == ord('l'):
                frames_shift = 100
            elif key == 32:  # space
                snapshot(pose_detector.last_image, pose_detect_dict)
            elif key != -1:
                print(f'key {key}')

            length = int(img_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = int(img_stream.get(cv2.CAP_PROP_POS_FRAMES))

            if frames_shift != 0:
                step = True
                current_frame += frames_shift -1
                current_frame = max(0,current_frame)
                current_frame = min(length, current_frame)
                frames_shift = 0
                img_stream.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                #print(current_frame)

            #print(f'{current_frame}/\t\t{length}')

    img_stream.release()





if __name__ == '__main__':
    run()
