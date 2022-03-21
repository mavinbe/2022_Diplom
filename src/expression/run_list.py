import mediapipe as mp
import numpy as np

from yolov5.utils.torch_utils import time_sync

#from app.run_pipeline import determ_position_by_landmark_from_pose_detection
from modules.pysical_models.new_position_max_speed_constrained import NewPositionMaxSpeedConstrained

PoseLandmark = mp.solutions.pose.PoseLandmark

def determ_position_by_landmark_from_pose_detection(pose_detect_dict, landmark):
    if landmark in pose_detect_dict:
        return pose_detect_dict[landmark]['x'], pose_detect_dict[landmark]['y']

class CuePoint:
    pass

class Pause(CuePoint):
    def __init__(self, seconds):
        self.seconds = seconds
        self.start_time = None

    def start(self, start_time):
        self.start_time = start_time

    def is_finished(self, time):
        print(time - self.start_time)
        return time - self.start_time >= self.seconds


class LandmarkTarget(CuePoint):
    def __init__(self, target):
        self.target = target
        self.position_model = None

    def start(self, start_time, start_position):
        self.position_model = NewPositionMaxSpeedConstrained(start_time, start_position, 240)

    def is_finished(self, pose_detect_dict_in_global):
        target_position = determ_position_by_landmark_from_pose_detection(pose_detect_dict_in_global,
                                                                  self.target)
        current_position = self.position_model.get_position()
        print((target_position, current_position))
        if target_position is None or current_position is None:
            return False
        target_position = np.array(target_position)
        current_position = np.array(current_position)
        diff = target_position - current_position
        magnitude = np.linalg.norm(diff)

        return magnitude < 5


def run_list():
    return [
        Pause(1.5),
        LandmarkTarget(PoseLandmark.NOSE),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.RIGHT_SHOULDER),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.RIGHT_HIP),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.RIGHT_KNEE),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.RIGHT_ANKLE),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.LEFT_ANKLE),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.LEFT_KNEE),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.LEFT_HIP),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.LEFT_SHOULDER),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.NOSE),
        Pause(3),
        LandmarkTarget(PoseLandmark.LEFT_THUMB),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.LEFT_INDEX),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.LEFT_PINKY),
        Pause(0.6),
        LandmarkTarget(PoseLandmark.RIGHT_PINKY),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.RIGHT_INDEX),
        Pause(0.3),
        LandmarkTarget(PoseLandmark.RIGHT_THUMB),
        Pause(10),

    ]