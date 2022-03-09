import cv2
import numpy as np
from utils.general import check_imshow
import unittest
from parameterized import parameterized

from yolov5.utils.torch_utils import time_sync


class NewPositionMaxSpeedConstrained:
    def __init__(self, time, position, max_velocity, show_vid=False):
        # Check if environment supports image displays
        if show_vid:
            show_vid = check_imshow()
        self.show_vid = show_vid

        self.current_time = time

        self.position_bounds = (0, 1920, 0, 1080)
        self.max_velocity = max_velocity
        #self.max_acceleration = 300

        self.position = position
        ##self.acceleration = (0, 0)

    def get_position(self):
        return self.position

    def move_to_target(self, target, new_time):
        time_delta = self._calculate_time_delta(new_time)
        new_position = self.calculate_new_position(target, time_delta)
        self.position = new_position

    def calculate_new_position(self, target, time_delta):
        return self.position + self.calculate_real_position_delta(self.position, self.max_velocity, target, time_delta)

    def _calculate_time_delta(self, new_time):
        return new_time - self.current_time

    @staticmethod
    def calculate_real_position_delta(position, max_velocity, target, time_delta):
        total_position_delta = target - position
        direction_norm = abs(total_position_delta) / total_position_delta
        calculated_position_delta = NewPositionMaxSpeedConstrained.calculate_position_delta(max_velocity * direction_norm, time_delta)

        real_position_delta = None
        if total_position_delta == 0:
            real_position_delta = 0
        elif total_position_delta > 0:
            real_position_delta = min(total_position_delta, calculated_position_delta)
        elif total_position_delta < 0:
            real_position_delta = max(total_position_delta, calculated_position_delta)
        return real_position_delta


    @staticmethod
    def calculate_position_delta(velocity, time_delta):
        return velocity * time_delta


class NewPositionMaxSpeedConstrainedTest(unittest.TestCase):

    def testInit(self):
        start_time = 0
        sut = NewPositionMaxSpeedConstrained(start_time, 100, 120)
        self.assertEqual(sut.get_position(), 100)

    @parameterized.expand([
        [0,     120,    100,    100],
        [0,     80,     100,    80],
        [200,   120,    100,    -100],
        [200,   80,     100,    -80],
    ])
    def test_calculate_real_position_delta(self, start_position, max_velocity, target_position, must_position_delta):
        time_delta = 1

        real_position_delta = NewPositionMaxSpeedConstrained.calculate_real_position_delta(start_position, max_velocity, target_position, time_delta)

        self.assertEqual(real_position_delta, must_position_delta)

    @parameterized.expand([
        [0,     120,    100,    100],
        [0,     80,     100,    80],
        [200,   120,    100,    100],
        [200,   80,     100,    120],
    ])
    def test_calculate_real_position(self, start_position, max_velocity, target_position, must_position):
        start_time = 0
        time_delta = 1
        sut = NewPositionMaxSpeedConstrained(start_time, start_position, max_velocity)

        real_position = sut.calculate_new_position(target_position, time_delta)

        self.assertEqual(real_position, must_position)


if __name__ == '__main__':
    unittest.main()