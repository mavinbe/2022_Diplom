import cv2
import mediapipe as mp
import numpy as np

from modules.object_tracker import ObjectTracker
from modules.pose_detector import PoseDetector
from utils.general import LOGGER
from modules.pysical_models.new_position_max_speed_constrained import NewPositionMaxSpeedConstrained
from yolov5.utils.torch_utils import time_sync

PoseLandmark = mp.solutions.pose.PoseLandmark


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
    try:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    except:

        print(img.shape)

    return img


def determ_position_by_landmark_from_pose_detection(pose_detect_dict, landmark):
    if landmark in pose_detect_dict:
        return pose_detect_dict[landmark]['x'], pose_detect_dict[landmark]['y']


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


def run(handle_image):
    with PoseDetector(show_vid=False) as pose_detector:

        object_tracker = ObjectTracker(show_vid=False)
        frame_count = 0
        img_stream = cv2.VideoCapture("/home/mavinbe/2021_Diplom/2022_Diplom/data/05_20211102141647/output014.mp4")
        height, width = determ_dimensions_of_video(img_stream)
        position_model = NewPositionMaxSpeedConstrained(time_sync(), np.asarray((int(width / 2), int(height / 2))), 20)

        while img_stream.isOpened():

            t_start = time_sync()

            # t_read_image
            image, success = read_frame_till_x(img_stream, frame_count, 1600)
            frame_count += 1
            if not success:
                continue
            t_read_image = time_sync()

            # t_object_track
            object_detection_dict = object_tracker.inference_frame(image)
            if len(object_detection_dict) == 0:
                print("No Objects Detected")
                continue
            t_object_track = time_sync()

            # t_pose_detect
            pose_detect_dict, pose_detect_dict_in_global = inference_pose(pose_detector, image, object_detection_dict,
                                                                          calculate_newest_track_id)
            t_pose_detect = time_sync()

            # t_post
            target_position = determ_position_by_landmark_from_pose_detection(pose_detect_dict_in_global,
                                                                              PoseLandmark.NOSE)
            if target_position is None:
                print("No Landmark found " + str(PoseLandmark.NOSE))
                continue

            position_model.move_to_target(target_position, time_sync())
            target_box = static_zoom_target_box(image.shape, 20, position_model.get_position())
            image = zoom(image, target_box)
            t_post = time_sync()

            # t_handle_image
            handle_image(image)

            t_handle_image = time_sync()

            LOGGER.info(
                f'frame_count {frame_count} DONE on hole :({(t_handle_image - t_start) * 1000:.3f}ms)    read_image:({(t_read_image - t_start) * 1000:.3f}ms), object_track:({(t_object_track - t_read_image) * 1000:.3f}ms), pose_detect:({(t_pose_detect - t_object_track) * 1000:.3f}ms), post:({(t_post - t_pose_detect) * 1000:.3f}ms), handle_image:({(t_handle_image - t_post) * 1000:.3f}ms)')
    img_stream.release()


def determ_dimensions_of_video(img_stream):
    current_frame_id = img_stream.get(cv2.CAP_PROP_POS_FRAMES)
    _, init_frame = img_stream.read()
    img_stream.set(cv2.CAP_PROP_POS_FRAMES, current_frame_id)
    height, width, _ = init_frame.shape
    return height, width


def inference_pose(pose_detector, image, object_detection_dict, calculate_track_id_strategy):
    track_id_to_track = calculate_track_id_strategy(object_detection_dict)
    detection_which_to_pose_detect = object_detection_dict[track_id_to_track]

    cropped_image = image[detection_which_to_pose_detect[1]:detection_which_to_pose_detect[3],
                    detection_which_to_pose_detect[0]:detection_which_to_pose_detect[2]]
    pose_detect_dict = pose_detector.inference_frame(cropped_image)
    pose_detect_dict_in_global = translate_local_to_global_coords(pose_detect_dict,
                                                                  detection_which_to_pose_detect[0],
                                                                  detection_which_to_pose_detect[1])
    return pose_detect_dict, pose_detect_dict_in_global


def read_frame_till_x(img_stream, frame_count, x):
    if frame_count >= x:
        exit()
    success, image = img_stream.read()
    if not success:
        print("Ignoring empty camera frame.")
    return image, success


def show_image(image):
    cv2.imshow("asd", image)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        exit(0)


if __name__ == '__main__':
    run(show_image)
