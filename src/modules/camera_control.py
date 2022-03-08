import cv2
import numpy as np
from utils.general import check_imshow
from yolov5.utils.torch_utils import time_sync


class CameraControl:
    def __init__(self, show_vid=False):
        # Check if environment supports image displays
        if show_vid:
            show_vid = check_imshow()
        self.show_vid = show_vid

        self.current_time = time_sync()
        self.position = (x,y)
        self.velocity = (vx,vy)
        self.acceleration = (ax, ay)

    def inference_frame(self, image):
        time_diff = time_sync()


if __name__ == '__main__':
