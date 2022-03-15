import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

from modules.object_tracker import ObjectTracker
from modules.pose_detector import PoseDetector
from utils.general import LOGGER
from modules.pysical_models.new_position_max_speed_constrained import NewPositionMaxSpeedConstrained
from yolov5.utils.torch_utils import time_sync
# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))

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


class PoseDetectorPool:
    def __init__(self):
        self.pool = []
        self.pool_map = {}

        for i in range(10):
            self.pool.append(PoseDetector(show_vid=False))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for detector in self.pool:
            detector.__exit__(exc_type, exc_val, exc_tb)

    def get(self, key):
        if key not in self.pool_map.keys():
            self.pool_map[key] = self._get_first_free_detector()

        return self.pool_map[key]

    def _get_first_free_detector(self):
        for detector in self.pool:
            if detector not in self.pool_map:
                return detector
        raise RuntimeError("PoseDetectorPool is exhausted")

def handle_pose_detect_list(image, object_detection_dict, pose_detector_pool, t):
    pose_detect_dict_in_global_dict = {}
    #print(object_detection_dict)
    # for key in object_detection_dict:
    #     # print(key)
    #     # print(pose_detector_pool.get(key))
    #     pose_detect_dict, pose_detect_dict_in_global = inference_pose(pose_detector_pool.get(key), image,
    #                                                                   object_detection_dict,
    #                                                                   key)
    #     pose_detect_dict_in_global_dict[key] = pose_detect_dict_in_global
    key = calculate_newest_track_id(object_detection_dict)
    _, pose_detect_dict_in_global_dict[key] = inference_pose(pose_detector_pool.get(key), image,
                                                             object_detection_dict,
                                                             key)
    t["pose_detect"] = time_sync()
    t["pose_detect_count"] = len(pose_detect_dict_in_global_dict)
    return pose_detect_dict_in_global_dict


def run(handle_image, serialize=True):
    with PoseDetectorPool() as pose_detector_pool:

        object_tracker = ObjectTracker(show_vid=False)
        frame_count = 0
        img_stream = cv2.VideoCapture(ROOT_DIR+"/data/05_20211102141647/output014.mp4")
        height, width = determ_dimensions_of_video(img_stream)
        position_model = NewPositionMaxSpeedConstrained(time_sync(), np.asarray((int(width / 2), int(height / 2))), 20)

        serialize_path = create_serialize_path() if serialize else None
        serialize_store = {} if serialize_path else None

        while img_stream.isOpened():
            current_time = time_sync()
            t = {"start": current_time, "read_image": None, "object_track": None, "pose_detect": None, "pose_detect_count": 0, "post": None, "handle_image": None}
            frame_count += 1
            serialize_store[frame_count] = {}
            try:
                # t_read_image
                image = handle_read_image(frame_count, img_stream, t)

                # t_object_track
                object_detection_dict, confirmed_id_list = handle_object_track(image, object_tracker, t)
                serialize_store[frame_count]["object_track"] = object_detection_dict

                # t_pose_detect
                pose_detect_dict_in_global = None
                #pose_detect_dict_in_global = handle_pose_detect(image, object_detection_dict, pose_detector, t)

                pose_detect_dict_in_global_dict = handle_pose_detect_list(image, object_detection_dict, pose_detector_pool, t)
                pose_detect_dict_in_global = pose_detect_dict_in_global_dict[calculate_newest_track_id(
                                                                      object_detection_dict)]

                # t_post
                image = handle_post(image, pose_detect_dict_in_global, position_model, t)

                # t_handle_image
                handle_image(image, t)

                LOGGER.info(
                    f'frame_count {frame_count} DONE on hole: \t({(t["handle_image"] - t["start"]) * 1000:.2f}ms)\tread_image:({(t["read_image"] - t["start"]) * 1000:.2f}ms)\tobject_track:({(t["object_track"] - t["read_image"]) * 1000:.2f}ms)\tpose_detect({t["pose_detect_count"]}):({(t["pose_detect"] - t["object_track"]) * 1000:.2f}ms)\tpost:({(t["post"] - t["pose_detect"]) * 1000:.2f}ms)\thandle_image:({(t["handle_image"] - t["post"]) * 1000:.2f}ms)')


            except Warning as warn:
                #print(str(warn))
                for key in t.keys():
                    if t[key] is None:
                        t[key] = time_sync()
                LOGGER.info(
                    f'frame_count {frame_count} DONE on hole: \t({(time_sync() - t["start"]) * 1000:.2f}ms)\tread_image:({(t["read_image"] - t["start"]) * 1000:.2f}ms)\tobject_track:({(t["object_track"] - t["read_image"]) * 1000:.2f}ms)\tpose_detect({t["pose_detect_count"]}):({(t["pose_detect"] - t["object_track"]) * 1000:.2f}ms)\tpost:({(t["post"] - t["pose_detect"]) * 1000:.2f}ms)\thandle_image:({(t["handle_image"] - t["post"]) * 1000:.2f}ms)\t--- {warn}')

                continue
            #finally:
                #print(t)

        write_detection(frame_count, object_detection_dict, serialize_path)
        img_stream.release()


def handle_read_image(frame_count, img_stream, t):
    image, success = read_frame_till_x(img_stream, frame_count, 1600)
    t["read_image"] = time_sync()
    if not success:
        raise Warning("No Frame")
    return image


def handle_object_track(image, object_tracker, t):
    object_detection_dict, confirmed_id_list = object_tracker.inference_frame(image)
    t["object_track"] = time_sync()
    if len(object_detection_dict) == 0:
        raise Warning("No Objects Detected")
    return object_detection_dict, confirmed_id_list


def handle_pose_detect(image, object_detection_dict, pose_detector, t):
    pose_detect_dict, pose_detect_dict_in_global = inference_pose(pose_detector, image,
                                                                  object_detection_dict,
                                                                  calculate_newest_track_id(
                                                                      object_detection_dict))
    t["pose_detect"] = time_sync()
    return pose_detect_dict_in_global


def handle_post(image, pose_detect_dict_in_global, position_model, t):
    target_position = determ_position_by_landmark_from_pose_detection(pose_detect_dict_in_global,
                                                                      PoseLandmark.NOSE)
    if target_position is None:
        t["post"] = time_sync()
        raise Warning("No Landmark found " + str(PoseLandmark.NOSE))
    position_model.move_to_target(target_position, time_sync())
    target_box = static_zoom_target_box(image.shape, 20, position_model.get_position())
    image = zoom(image, target_box)
    t["post"] = time_sync()
    return image


def write_detection(frame_count, object_detection_dict, serialize_path):
    if serialize_path:
        serialize_filename = str(frame_count) + '.p'
        serialize_filepath = os.path.join(serialize_path, serialize_filename)
        with open(serialize_filepath, 'wb') as file:
            pickle.dump(object_detection_dict, file)


def create_serialize_path():
    serialize_path = os.path.join(ROOT_DIR, "data/05_20211102141647/output014.serial/")
    print(serialize_path)
    if not os.path.exists(serialize_path):
        os.makedirs(serialize_path)
    return serialize_path


def determ_dimensions_of_video(img_stream):
    current_frame_id = img_stream.get(cv2.CAP_PROP_POS_FRAMES)
    _, init_frame = img_stream.read()
    img_stream.set(cv2.CAP_PROP_POS_FRAMES, current_frame_id)
    height, width, _ = init_frame.shape
    return height, width


def inference_pose(pose_detector, image, object_detection_dict, track_id_to_track):
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


def show_image(image, t):
    cv2.imshow("asd", image)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        exit(0)
    t["handle_image"] = time_sync()


if __name__ == '__main__':
    run(show_image)
