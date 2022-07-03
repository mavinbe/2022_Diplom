import mediapipe as mp
import enum

import numpy as np

PoseLandmark = mp.solutions.pose.PoseLandmark


class VirtualPoseLandmark(enum.IntEnum):
    """The 33 pose landmarks."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
    MID_HIP = 33
    MID_SHOULDER = 34
    STOMACHE = 35
    BREAST = 36
    CROTCH = 37

    def determ_position_by_Vlandmark_from_pose_detection(self, pose_detect_dict):
        if self < 33:
            return determ_position_by_landmark_from_pose_detection(pose_detect_dict, self)
        elif self in [33, 34, 35, 36, 37]:
            try:
                # if all_y_in_x([33, 34, 35, 36, 37], pose_detect_dict):

                mid_hip = determ_half_way(VirtualPoseLandmark.LEFT_HIP.determ_position_by_Vlandmark_from_pose_detection(pose_detect_dict), VirtualPoseLandmark.RIGHT_HIP.determ_position_by_Vlandmark_from_pose_detection(pose_detect_dict))
                mid_shoulder = determ_half_way(
                    VirtualPoseLandmark.LEFT_SHOULDER.determ_position_by_Vlandmark_from_pose_detection(pose_detect_dict),
                    VirtualPoseLandmark.RIGHT_SHOULDER.determ_position_by_Vlandmark_from_pose_detection(pose_detect_dict))
                if self == self.MID_HIP:
                    return mid_hip
                elif self == self.MID_SHOULDER:
                    return mid_shoulder
                elif self == self.STOMACHE:
                    return determ_half_Xway(mid_hip, mid_shoulder, 0.2)
                elif self == self.BREAST:
                    return determ_half_Xway(mid_hip, mid_shoulder, 0.72)
                elif self == self.CROTCH:
                    return determ_half_Xway(mid_hip, mid_shoulder, -0.2)
            except Exception as e:
                return None

def all_y_in_x(y, x):
    return all(i in x for i in y)

def determ_position_by_landmark_from_pose_detection(pose_detect_dict, landmark):
    if landmark in pose_detect_dict:
        return pose_detect_dict[landmark]['x'], pose_detect_dict[landmark]['y']

def determ_half_way(a, b):
    return determ_half_Xway(a, b, 0.5)

def determ_half_Xway(a, b, X):
    if a is None or b is None:
        raise Exception("determ_half_Xway: a or b is None")
    a = np.array(a)
    b = np.array(b)
    calc = (b-a) * X + a
    return (int(calc[0]),int(calc[1]))

class HalfwayLandmark():
    def __init__(self, pose_1, pose_2):
        self.pose_1 = pose_1
        self.pose_2 = pose_2
