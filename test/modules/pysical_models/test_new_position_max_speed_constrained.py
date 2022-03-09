import numpy as np
import unittest
from parameterized import parameterized
from modules.pysical_models.new_position_max_speed_constrained import NewPositionMaxSpeedConstrained


class TestNewPositionMaxSpeedConstrainedTest(unittest.TestCase):

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
    def test_calculate_real_position_delta_1D(self, start_position, max_velocity, target_position, must_position_delta):
        start_position = np.array(start_position)
        target_position = np.array(target_position)
        must_position_delta = np.array(must_position_delta)
        time_delta = 1

        real_position_delta = NewPositionMaxSpeedConstrained.calculate_real_position_delta_1D(start_position, max_velocity, target_position,
                                                                                              time_delta)

        self.assertEqual(real_position_delta, must_position_delta)

    @parameterized.expand([
        [[0],     120,    [100],    [100]],
        [[0],     80,     [100],    [80]],
        [[200],   120,    [100],    [100]],
        [[200],   80,     [100],    [120]],
    ])
    def test_calculate_real_position_1D(self, start_position, max_velocity, target_position, must_position):
        start_position = np.array(start_position)
        target_position = np.array(target_position)
        must_position = np.array(must_position)
        start_time = 0
        time_delta = 1
        sut = NewPositionMaxSpeedConstrained(start_time, start_position, max_velocity)
        real_position = sut.calculate_new_position(target_position, time_delta)

        self.assertEqual(real_position, must_position)

    @parameterized.expand([
        [(0, 0),        120,        (100, 100),         (100, 100)],
        [(0, 0),        80,         (100, 100),         (80, 80)],
        [(200, 400),    120,        (100, 100),         (100, 280)],
        [(200, 400),    80,         (100, 100),         (120, 320)],
    ])
    def test_calculate_real_position_2D(self, start_position, max_velocity, target_position, must_position):
        start_position = np.array(start_position)
        target_position = np.array(target_position)
        must_position = np.array(must_position)

        start_time = 0
        time_delta = 1
        sut = NewPositionMaxSpeedConstrained(start_time, start_position, max_velocity)

        real_position = sut.calculate_new_position(target_position, time_delta)
        print(real_position)

        print(must_position)
        np.testing.assert_array_equal(real_position, must_position)


if __name__ == '__main__':
    unittest.main()