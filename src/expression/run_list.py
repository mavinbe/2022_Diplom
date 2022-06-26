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
        if self.is_started():
            raise RuntimeError("Is already started.")
        self.start_time = start_time

    def is_started(self):
        return self.start_time is not None

    def is_finished(self, time):
        if not self.is_started():
            print("if not self.is_started():")
            return False
        return time - self.start_time >= self.seconds


class PositionTarget(CuePoint):
    def __init__(self, target, target_zoom=None, movement_v_coefficient=4, zoom_v_coefficient=4, after_finished=None):
        self.target = target
        self.target_zoom = target_zoom
        if target_zoom is not None:
            self.target_zoom = np.array([target_zoom])
        self.movement_v_coefficient = movement_v_coefficient
        self.zoom_v_coefficient = zoom_v_coefficient
        self.after_finished = after_finished

        self.position_model = None
        self.zoom_model = None


    def start(self, start_time, start_position, start_zoom):
        self.position_model = NewPositionMaxSpeedConstrained(start_time, start_position, self.movement_v_coefficient, 480)
        self.zoom_model = NewPositionMaxSpeedConstrained(
                        time_sync(), start_zoom, self.zoom_v_coefficient, 5)

    def is_finished(self, pose_detect_dict_in_global):

        if self.after_finished is None:
            return self.self_is_finished(pose_detect_dict_in_global)
        else:
            return self.self_is_finished(pose_detect_dict_in_global) and self.after_finished_is_finished()

        # if not self.self_is_finished(pose_detect_dict_in_global):
        #     return False
        #
        # if self.after_finished is None:
        #    return False
        # else:
        #     return self.after_finished_is_finished()

    def after_finished_is_finished(self):
        if not self.after_finished.is_started():
            self.after_finished.start(time_sync())

        return self.after_finished.is_finished(time_sync())

    def self_is_finished(self, pose_detect_dict_in_global):
        return self._is_position_finished(pose_detect_dict_in_global) and self._is_zoom_finished(pose_detect_dict_in_global)

    def _is_position_finished(self, pose_detect_dict_in_global):
        target_position = self.determ_position(pose_detect_dict_in_global)
        #print(target_position)
        current_position = self.position_model.get_position()
        if target_position is None or current_position is None:
            return False
        target_position = np.array(target_position)
        current_position = np.array(current_position)
        diff = target_position - current_position
        magnitude = np.linalg.norm(diff)

        return magnitude < 5

    def determ_position(self, pose_detect_dict_in_global):
        return self.target

    def _is_zoom_finished(self, pose_detect_dict_in_global):
        if self.target_zoom is None:
            return True

        diff = self.target_zoom - np.array([self.zoom_model.get_position()])
        magnitude = np.linalg.norm(diff)
        #print(magnitude)
        return magnitude < 0.1

class LandmarkTarget(PositionTarget):
    def determ_position(self, pose_detect_dict_in_global):
        return determ_position_by_landmark_from_pose_detection(pose_detect_dict_in_global,
                                                               self.target)

def run_list_1():
    return [
        Pause(3),
        LandmarkTarget(PoseLandmark.NOSE, target_zoom=8, zoom_v_coefficient=1, after_finished=Pause(10)),
        LandmarkTarget(PoseLandmark.NOSE, target_zoom=8, zoom_v_coefficient=1, after_finished=Pause(1)),
        LandmarkTarget(PoseLandmark.RIGHT_ANKLE,
                       after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.LEFT_ANKLE,
                       after_finished=Pause(0.3)),
        LandmarkTarget(PoseLandmark.NOSE,
                       after_finished=Pause(0.1)),
        LandmarkTarget(PoseLandmark.NOSE, after_finished=Pause(5)),

        LandmarkTarget(PoseLandmark.NOSE, 15, after_finished=Pause(0.1)),
        LandmarkTarget(PoseLandmark.RIGHT_EYE_OUTER, 23, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.LEFT_EYE_OUTER, movement_v_coefficient=1, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.RIGHT_EYE_OUTER, movement_v_coefficient=1, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.LEFT_EYE_OUTER, movement_v_coefficient=1, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.NOSE, 15, after_finished=Pause(0.1)),
        LandmarkTarget(PoseLandmark.NOSE, 8, after_finished=Pause(7)),

        LandmarkTarget(PoseLandmark.RIGHT_THUMB, 14, after_finished=Pause(12)),
        LandmarkTarget(PoseLandmark.LEFT_THUMB, after_finished=Pause(12)),
        LandmarkTarget(PoseLandmark.NOSE, 4,after_finished=Pause(15)),

        PositionTarget((623, 945), target_zoom=5, zoom_v_coefficient=1, after_finished=None),

    ]


def run_list_2():
    return [
        Pause(3),
        LandmarkTarget(PoseLandmark.NOSE, target_zoom=8, zoom_v_coefficient=1, after_finished=Pause(10)),
        LandmarkTarget(PoseLandmark.NOSE, target_zoom=8, zoom_v_coefficient=1, after_finished=Pause(1)),
        LandmarkTarget(PoseLandmark.RIGHT_ANKLE,
                       after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.LEFT_ANKLE,
                       after_finished=Pause(0.3)),
        LandmarkTarget(PoseLandmark.NOSE,
                       after_finished=Pause(0.1)),
        LandmarkTarget(PoseLandmark.NOSE, after_finished=Pause(5)),

        LandmarkTarget(PoseLandmark.RIGHT_THUMB, 14, after_finished=Pause(12)),
        LandmarkTarget(PoseLandmark.LEFT_THUMB, after_finished=Pause(12)),
        LandmarkTarget(PoseLandmark.NOSE, 8, after_finished=Pause(7)),

        LandmarkTarget(PoseLandmark.NOSE, 15, after_finished=Pause(0.1)),
        LandmarkTarget(PoseLandmark.RIGHT_EYE_OUTER, 23, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.LEFT_EYE_OUTER, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.RIGHT_EYE_OUTER, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.LEFT_EYE_OUTER, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.RIGHT_EYE_OUTER, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.LEFT_EYE_OUTER, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.RIGHT_EYE_OUTER, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.LEFT_EYE_OUTER, movement_v_coefficient=6, after_finished=Pause(0.01)),
        LandmarkTarget(PoseLandmark.NOSE, 15, after_finished=Pause(0.1)),
        LandmarkTarget(PoseLandmark.NOSE, 4, after_finished=Pause(15)),

        PositionTarget((1417, 633), target_zoom=8, zoom_v_coefficient=1, after_finished=None),

    ]