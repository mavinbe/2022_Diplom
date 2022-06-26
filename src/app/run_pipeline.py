import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import pickle
import atexit

import multiprocessing as multiP

from expression.run_list import run_list_1, run_list_2,  Pause, CuePoint, LandmarkTarget, PositionTarget
from media.SampleDataProvider import get_sample_video_specification_by_key, get_video_stream_provider
from media.VideoGStreamerProvider import VideoGStreamerProvider
from media.VideoStreamProvider import VideoStreamProvider
from modules.object_tracker import ObjectTracker
from modules.pose_detector import PoseDetector
from utils.general import LOGGER
from modules.pysical_models.new_position_max_speed_constrained import NewPositionMaxSpeedConstrained
from yolov5.utils.torch_utils import time_sync
# Root directory of the project
#from network.ImageSendingProvider import ImageSendingProvider
import screeninfo

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))

PoseLandmark = mp.solutions.pose.PoseLandmark

VEBOSE_MODE = False


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
    #image_original_shape = img.shape
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

        for i in range(1):
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

    def get_first_detector(self):
        return self.pool[0]

    def _get_first_free_detector(self):
        for detector in self.pool:
            if detector not in self.pool_map.values():
                return detector
        raise RuntimeError("PoseDetectorPool is exhausted")

def handle_pose_detect_list(image, object_detection_dict, pose_detector_pool, t):
    pose_detect_dict_in_global_dict = {}
    #print(object_detection_dict)
    for key in object_detection_dict:
        # print(key)
        # print(pose_detector_pool.get(key))
        # pose_detect_dict, pose_detect_dict_in_global = inference_pose(pose_detector_pool.get(key), image,
        #                                                               object_detection_dict,
        #                                                               key)
        pose_detect_dict, pose_detect_dict_in_global = inference_pose(pose_detector_pool.get_first_detector(), image,
                                                                     object_detection_dict,
                                                                     key)
        pose_detect_dict_in_global_dict[key] = pose_detect_dict_in_global

    t["pose_detect"] = time_sync()
    t["pose_detect_count"] = len(pose_detect_dict_in_global_dict)
    return pose_detect_dict_in_global_dict


def print_data_to_image(image, state, position):
    text = str(state)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2)

    return image


def draw_box_to_image(image, in_room_zone, param):
    return cv2.rectangle(image, in_room_zone[0], in_room_zone[1], (0,0,255), thickness=2, lineType=cv2.LINE_AA)


def determ_is_empty_room(object_detection_dict, confirmed_id_list, exit_zone):
    # print(len(confirmed_id_list))
    # print((object_detection_dict, confirmed_id_list, exit_zone))
    return len(confirmed_id_list) == 0

def determ_is_following():
    raise NotImplementedError


def determ_are_persons_left():
    raise NotImplementedError


def print_detections(image, object_detection_dict):
    for key in object_detection_dict.keys():
        box = object_detection_dict[key]
        color = (255, 0, 0)
        lw = 2

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if key:
            key = str(key)
            tf = max(lw - 1, 1)
            w, h = cv2.getTextSize(key, 0, fontScale=lw / 3,  thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, key, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, (255,255,255),
                        thickness=tf, lineType=cv2.LINE_AA)
    return image


def x_is_in_range(x, range):
    return range[0] < x < range[1]


def is_detection_in_zone(det, exit_zone):

    return x_is_in_range(det[0], exit_zone[0]) and x_is_in_range(det[2], exit_zone[0]) and x_is_in_range(det[1], exit_zone[1]) and x_is_in_range(det[3], exit_zone[1])


def remove_old_persons(persons_not_in_exit_zone):
    keys = list(persons_not_in_exit_zone.keys())
    for key in keys:
        if persons_not_in_exit_zone[key]['last_time'] + 6 < time_sync():
            del persons_not_in_exit_zone[key]
    return persons_not_in_exit_zone

def update_persons_not_in_exit_zone(persons_not_in_exit_zone, exit_zone, object_detection_dict):
    persons_not_in_exit_zone = remove_old_persons(persons_not_in_exit_zone)

    for key in object_detection_dict.keys():
        det = object_detection_dict[key]
        if is_detection_in_zone(det, exit_zone):
            if key in persons_not_in_exit_zone.keys():
                del persons_not_in_exit_zone[key]
        else:
            persons_not_in_exit_zone[key] = {'last_time': time_sync()}

    return persons_not_in_exit_zone


def run(handle_image, img_stream_data, sink_ip, track_highest, run_list, out_queue=None, in_queue=None):
    with PoseDetectorPool() as pose_detector_pool:

        img_stream = None
        if img_stream_data['type'] == 'cam':
            img_stream = VideoGStreamerProvider('rtspsrc location='+img_stream_data['url']+' latency=1 !  rtph264depay ! avdec_h264 !  videoconvert ! videoscale ! appsink')
        elif img_stream_data['type'] == 'file':
            img_stream = get_video_stream_provider(ROOT_DIR, img_stream_data['video_specification'])
        print(img_stream)
        object_tracker = ObjectTracker(show_vid=False)
        frame_count = 0
        #img_stream = cv2.VideoCapture(ROOT_DIR+"/data/05_20211102141647/output014.mp4")
        #img_stream = cv2.VideoCapture("rtsp://malte:diplom@192.168.0.105:554//h264Preview_06_main")
        #height, width = determ_dimensions_of_video(img_stream)
        #img_stream = VideoStreamProvider(ROOT_DIR + "/data/05_20211102141647/output017.mp4", play_back_speed=0.4)
        #img_stream = VideoStreamProvider("rtsp://malte:diplom@192.168.0.110:554//h264Preview_06_main")
        # img_stream = VideoGStreamerProvider(
        #     'rtspsrc location='+cam_url+' latency=1 !  rtph264depay ! avdec_h264 !  videoconvert ! videoscale ! appsink')

        atexit.register(img_stream.release)


        screen = screeninfo.get_monitors()[0]
        print(F"Screen {screen}")
        #cv2.namedWindow("asd", cv2.WND_PROP_FULLSCREEN)
        #cv2.moveWindow("asd", int(screen.x - 1), int(screen.y - 1))
        #cv2.setWindowProperty("asd", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        height, width = (1080, 1920)
        print((width, height))
        framerate = 25
        send_out_1 = cv2.VideoWriter(
            "appsrc ! videoconvert ! video/x-raw,format=I420 ! jpegenc ! rtpjpegpay ! rtpstreampay ! udpsink host="+sink_ip+" port=7001",
            cv2.CAP_GSTREAMER, 0, framerate, (width, height), True)
        atexit.register(send_out_1.release)

        run_item = None
        _run_list = run_list()
        current_box = None
        current_is_blury = False
        current_position = None
        pose_to_follow = None
        current_zoom = np.array([1])
        persons_not_in_exit_zone = {}
        while True:
            # is in_queue is set wait till get returns a value
            wait_for_sync(in_queue, out_queue)

            current_time = time_sync()
            t = {"start": current_time, "read_image": None, "object_track": None, "pose_detect": None, "pose_detect_count": 0, "camera_movement": None, "handle_image": None}
            frame_count += 1
            try:
                # t_read_image
                original_image = handle_read_image(frame_count, img_stream, t)

                # crop and zoom for beamer fit at probeaugbau
                original_image = crop_and_zoom_for_beamer_fit(height, original_image, width)

                # create clone
                image = original_image

                #raise Warning("asdf")
                # t_object_track
                object_detection_dict, confirmed_id_list = handle_object_track(image, object_tracker, t)
                #print((confirmed_id_list,object_detection_dict))
                                    # x1               y1                      x2                        y2
                exit_zone = ((int(width * 13 / 20), int(height * 2 / 40)), (int(width * 15 / 20), int(height * 17 / 40)))
                # exit_zone = ((0, int(height * 19 / 40)), (int(width / 6), int(height * 17 / 20)))
                to_do = []


                persons_not_in_exit_zone = update_persons_not_in_exit_zone(persons_not_in_exit_zone, exit_zone, object_detection_dict)



                if determ_is_empty_room(object_detection_dict, confirmed_id_list, persons_not_in_exit_zone):
                    to_do.append("reset")
                    print("------------------------ RESET ")

                    run_item = None
                    _run_list = run_list()

                    current_box = (0, original_image.shape[0], 0, original_image.shape[1])
                    current_position = None
                    current_is_blury = False
                    pose_to_follow = None
                    current_zoom = np.array([1])
                    persons_not_in_exit_zone = {}

                # else:
                #     if determ_is_following():
                #         to_do.append("follow")
                #     else:
                #         if determ_are_persons_left():
                #             to_do.append("next_person")
                #         else:
                #             to_do.append("sleep")

                if VEBOSE_MODE:
                    image = print_detections(image, object_detection_dict)
                    image = print_data_to_image(image, to_do, (100, 100))
                    image = print_data_to_image(image, (frame_count, confirmed_id_list, object_detection_dict), (100, 500))
                    image = print_data_to_image(image, (persons_not_in_exit_zone), (100, 600))
                    image = draw_box_to_image(image, exit_zone, (100, 100))
                    image_with_hud = image



                # t_pose_detect
                pose_id_to_follow = None
                # print(object_detection_dict)
                if len(object_detection_dict) >= 1:


                    if track_highest:
                        pose_id_to_follow = max(confirmed_id_list)
                    else:
                        pose_id_to_follow = min(confirmed_id_list)
                    print("pose_id_to_follow")
                    print(pose_id_to_follow)
                    if pose_id_to_follow not in object_detection_dict:
                        run_item = None
                        _run_list = run_list()
                        pose_to_follow = None
                        raise Warning("pose_id_to_follow '" + str(pose_id_to_follow) + "' not in object_detection_dict '"+ str(object_detection_dict) +"'")
                    poses_to_detect = [pose_id_to_follow]

                    object_detection_dict_filtered = {your_key: object_detection_dict[your_key] for your_key in poses_to_detect}

                    pose_detect_dict_in_global = handle_pose_detect_list(image, object_detection_dict_filtered, pose_detector_pool, t)
                    pose_to_follow = pose_detect_dict_in_global[pose_id_to_follow]
                else:
                    raise Warning(
                        "object_detection_dict is empty ")

                if pose_to_follow:
                    _run_list, current_box, current_position, current_is_blury, current_zoom, height, image, pose_to_follow, run_item, t, width = run_animation(
                        _run_list, current_box, current_position, current_is_blury, current_zoom, height, image, pose_to_follow,
                        run_item, t, width)
                else:
                    raise Warning(
                        "pose_to_follow is "+str(pose_to_follow))
                # t_post

                if VEBOSE_MODE:
                    print("exit_zone")
                    print(exit_zone)
                    print("current_box")
                    print(current_box)

                    if current_box:
                        image_with_hud = draw_box_to_image(image_with_hud, ((current_box[2], current_box[0]),(current_box[3], current_box[1])), (100, 100))

                    image = image_with_hud

                # t_handle_image
                if current_is_blury is True:
                    image = cv2.blur(image, (10, 10))
                send_out_1.write(image)
                handle_image(image)
                t["handle_image"] = time_sync()
                # LOGGER.info(
                #     f'frame_count {frame_count} DONE on hole: \t({(t["handle_image"] - t["start"]) * 1000:.2f}ms)\tread_image:({(t["read_image"] - t["start"]) * 1000:.2f}ms)\tobject_track:({(t["object_track"] - t["read_image"]) * 1000:.2f}ms)\tpose_detect({t["pose_detect_count"]}):({(t["pose_detect"] - t["object_track"]) * 1000:.2f}ms) \tcamera_movement:({(t["camera_movement"] - t["pose_detect"]) * 1000:.2f}ms)\thandle_image:({(t["handle_image"] - t["camera_movement"]) * 1000:.2f}ms)')

            except Warning as warn:
                print("WARN: "+str(warn))
                for key in t.keys():
                    if t[key] is None:
                        t[key] = time_sync()
                if not VEBOSE_MODE:
                    image = zoom(image, current_box)


                if current_is_blury is True:
                    image = cv2.blur(image, (10, 10))
                send_out_1.write(image)
                handle_image(image)
                t["handle_image"] = time_sync()
                # LOGGER.info(
                #     f'frame_count {frame_count} DONE on hole: \t({(t["handle_image"] - t["start"]) * 1000:.2f}ms)\tread_image:({(t["read_image"] - t["start"]) * 1000:.2f}ms)\tobject_track:({(t["object_track"] - t["read_image"]) * 1000:.2f}ms)\tpose_detect({t["pose_detect_count"]}):({(t["pose_detect"] - t["object_track"]) * 1000:.2f}ms) \tcamera_movement:({(t["camera_movement"] - t["pose_detect"]) * 1000:.2f}ms)\thandle_image:({(t["handle_image"] - t["camera_movement"]) * 1000:.2f}ms)\t--- {warn}')

                continue
            #finally:
                #print(t)

        #write_detection(frame_count, object_detection_dict, serialize_path)
        img_stream.release()
        send_out_1.release()


def run_animation(_run_list, current_box, current_position, current_is_blury, current_zoom, height, image,
                  pose_to_follow, run_item, t, width):
    # start run_items
    if run_item is None:
        run_item = _run_list.pop(0)
        if isinstance(run_item, Pause):
            run_item.start(time_sync())
        if isinstance(run_item, LandmarkTarget):
            if current_position is None:
                current_position = determ_position_by_landmark_from_pose_detection(pose_to_follow,
                                                                                   run_item.target)
                print(current_position)
            run_item.start(time_sync(), current_position, current_zoom)
        elif isinstance(run_item, PositionTarget):
            if current_position is None:
                current_position = (int(width / 2), int(height / 2))
            run_item.start(time_sync(), current_position, current_zoom)
    if type(run_item) == PositionTarget:
        if run_item.self_is_finished(None):
            current_is_blury = True
    # clean up run_items
    if isinstance(run_item, Pause):
        if run_item.is_finished(time_sync()):
            run_item = None
    if isinstance(run_item, LandmarkTarget):
        if run_item.is_finished(pose_to_follow):
            run_item = None
    elif isinstance(run_item, PositionTarget):
        if run_item.is_finished(None):
            run_item = None
    # process run_item
    if isinstance(run_item, Pause):
        pass
        # print("Pause: ")
    if isinstance(run_item, LandmarkTarget):
        image, current_box, current_position, current_zoom = handle_camera_movement_with_LandmarkTarget(image,
                                                                                                        determ_position_by_landmark_from_pose_detection(
                                                                                                            pose_to_follow,
                                                                                                            run_item.target),
                                                                                                        run_item, t)
        # print("landmark: "+str(current_zoom))
    elif isinstance(run_item, PositionTarget):
        image, current_box, current_position, current_zoom = handle_camera_movement_with_LandmarkTarget(image,
                                                                                                        run_item.determ_position(
                                                                                                            None),
                                                                                                        run_item, t)

    else:
        if current_box:
            image = zoom(image, current_box)
        t["camera_movement"] = time_sync()
    return _run_list, current_box, current_position, current_is_blury, current_zoom, height, image, pose_to_follow, run_item, t, width


def crop_and_zoom_for_beamer_fit(height, original_image, width):
    original_image = original_image[383:, :]
    original_image = cv2.resize(original_image, (width, height))
    return original_image


def wait_for_sync(in_queue, out_queue):
    if in_queue:
        in_queue.get()
        # print(in_queue.get())
    if out_queue:
        out_queue.put('go')


def handle_read_image(frame_count, img_stream, t):
    image, success = read_frame_till_x(img_stream, frame_count, 1600)
    t["read_image"] = time_sync()
    if not success:
        raise Warning("No Frame")
    return image

# object_detection_dict is current detected id
# confirmed_id_list all ids that are not deleted yet
def handle_object_track(image, object_tracker, t):
    object_detection_dict, confirmed_id_list = object_tracker.inference_frame(image)
    t["object_track"] = time_sync()
    # if len(object_detection_dict) == 0:
    #     raise Warning("No Objects Detected")
    return object_detection_dict, confirmed_id_list

def handle_camera_movement_with_LandmarkTarget(image, target_position, landmark_target, t):
    if target_position is None:
        t["camera_movement"] = time_sync()
        raise Warning("No Landmark found " + str(PoseLandmark.NOSE))
    landmark_target.position_model.move_to_target(target_position, time_sync())
    current_zoom = None
    if landmark_target.target_zoom is not None:
        landmark_target.zoom_model.move_to_target(landmark_target.target_zoom, time_sync())
    current_zoom = landmark_target.zoom_model.get_position()[0]
    current_position = landmark_target.position_model.get_position()

    current_box = static_zoom_target_box(image.shape, current_zoom, current_position)
    image = zoom(image, current_box)
    t["camera_movement"] = time_sync()
    return image, current_box, np.array(current_position), np.array([current_zoom])


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
        pass
        #exit()
    success, image = img_stream.read()
    if not success:
        print("Ignoring empty camera frame.")
    return image, success

#image_sending_provider = ImageSendingProvider(server_port=5556)
def show_image(image):
    #cv2.imshow("asd", image)
    height, width, _ = image.shape


    # image = cv2.resize(image, (int(width/2), int(height/2)), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("asd", image)

    #print(image.shape)
    #image_sending_provider.send(image)

    if cv2.waitKey(1) == ord('q'):  # q to quit
        exit(0)



if __name__ == '__main__':
    sample_data = get_sample_video_specification_by_key("clara_simon_kira_me_hole")


    multiP.set_start_method('spawn')
    sync_queues = [multiP.Queue(), multiP.Queue()]
    # p_1 = multiP.Process(target=run, args=(show_image, {'type': 'cam', 'url': 'rtsp://malte:diplom@192.168.0.110:554//h264Preview_01_main'}, '192.168.0.101', False, run_list_1), kwargs={'in_queue': sync_queues[1], 'out_queue': sync_queues[0]})
    # p_1 = multiP.Process(target=run, args=(show_image, {'type': 'file', 'video_specification': sample_data[0]}, '192.168.0.101', False, run_list_1), kwargs={'in_queue': sync_queues[1], 'out_queue': sync_queues[0]})

    # p_1.start()

    # p_2 = multiP.Process(target=run, args=(show_image, {'type': 'cam', 'url': 'rtsp://malte:diplom@192.168.0.110:554//h264Preview_06_main'}, '192.168.0.102', False, run_list_2), kwargs={'in_queue': sync_queues[0], 'out_queue': sync_queues[1]})
    #p_2 = multiP.Process(target=run, args=(show_image, {'type': 'file', 'path': ROOT_DIR + "/data/2022_04_nice/06_20220421125959_part1_split_2.mp4", 'fps': 10.02, 'minute_to_start': 0}, '192.168.0.102', False, run_list_2), kwargs={'in_queue': sync_queues[0], 'out_queue': sync_queues[1]})


    p_2 = multiP.Process(target=run, args=(show_image, {'type': 'file', 'video_specification': sample_data[1]}, '192.168.0.102', False, run_list_2), kwargs={'in_queue': sync_queues[0], 'out_queue': sync_queues[0]})

    p_2.start()


    for out_queue in sync_queues:
        out_queue.put('go')

    # p_1.join()
    p_2.join()
