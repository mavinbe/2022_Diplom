import numpy as np
import unittest
from parameterized import parameterized
#from modules.pysical_models.movement_model import MovementModel
from interval import interval, inf, imath

from modules.pysical_models.lab.movement_prediction_model import MovementPredictionModel

class SymetricWithPlateu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movementModel = MovementPredictionModel(1, 0, 0, ([0, 2], [2, 4], [4, 6]))


    @parameterized.expand([
        [0, 1],
        [1, 1],
        [2, 1],
        [2.1, 0],
        [3, 0],
        [4, 0],
        [4.1, -1],
        [5, -1],
        [6, -1],
    ])
    def test_a_valid(self, t, a):
        sut = self.movementModel
        self.assertEqual(sut.a(t), a)

    @parameterized.expand([
        [0, np.nan],
        [1, 1],
        [2, np.nan],
        [2.1, 0],
        [3, 0],
        [4, np.nan],
        [4.1, -1],
        [5, -1],
        [6, np.nan],
    ])
    def test_a_for_plot_valid(self, t, a):
        sut = self.movementModel
        np.testing.assert_equal(sut.a_for_plot(t), a)

    @parameterized.expand([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 2],
        [4, 2],
        [5, 1],
        [6, 0],
    ])
    def test_v(self, t, v):
        sut = self.movementModel
        self.assertEqual(sut.v(t), v)

    @parameterized.expand([
        [0, 0],
        [1, 0.5],
        [2, 2],
        [3, 4],
        [4, 6],
        [5, 7.5],
        [6, 8],
    ])
    def test_s(self, t, v):
        sut = self.movementModel
        self.assertEqual(sut.s(t), v)




    @parameterized.expand([
        [-1], [6.1], [7]])
    def test_x_error_t_out_of_interval(self, t):
        sut = self.movementModel

        with self.assertRaises(RuntimeError):
            sut.a(t)

        with self.assertRaises(RuntimeError):
            sut.v(t)

        with self.assertRaises(RuntimeError):
            sut.s(t)

class InRunToZeroWithPlateu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movementModel = MovementPredictionModel(1, 1, 0, ([0, 1], [1, 3], [3, 5]))

    @parameterized.expand([
        [0, 1],
        [1, 1],
        [1.1, 0],
        [2, 0],
        [3, 0],
        [3.1, -1],
        [4, -1],
        [5, -1],
    ])
    def test_a_valid(self, t, a):
        sut = self.movementModel
        self.assertEqual(sut.a(t), a)

    @parameterized.expand([
        [0, np.nan],
        [1, np.nan],
        [1.1, 0],
        [2, 0],
        [3, np.nan],
        [3.1, -1],
        [4, -1],
        [5, np.nan],
    ])
    def test_a_for_plot_valid(self, t, a):
        sut = self.movementModel
        np.testing.assert_equal(sut.a_for_plot(t), a)

    @parameterized.expand([
        [0, 1],
        [1, 2],
        [2, 2],
        [3, 2],
        [4, 1],
        [5, 0],
    ])
    def test_v(self, t, v):
        sut = self.movementModel
        self.assertEqual(sut.v(t), v)

    @parameterized.expand([
        [0, 0],
        [1, 1.5],
        [2, 3.5],
        [3, 5.5],
        [4, 7],
        [5, 7.5],
    ])
    def test_s(self, t, v):
        sut = self.movementModel
        self.assertEqual(sut.s(t), v)

    @parameterized.expand([
        [-1], [6.1], [7]])
    def test_x_error_t_out_of_interval(self, t):
        sut = self.movementModel

        with self.assertRaises(RuntimeError):
            sut.a(t)

        with self.assertRaises(RuntimeError):
            sut.v(t)

        with self.assertRaises(RuntimeError):
            sut.s(t)

    #
    # @parameterized.expand([
    #     [0,     120,    100,    100],
    #     [0,     80,     100,    80],
    #     [200,   120,    100,    -100],
    #     [200,   80,     100,    -80],
    # ])
    # def test_calculate_real_position_delta_1D(self, start_position, max_velocity, target_position, must_position_delta):
    #     start_position = np.array(start_position)
    #     target_position = np.array(target_position)
    #     must_position_delta = np.array(must_position_delta)
    #     time_delta = 1
    #
    #     real_position_delta = NewPositionMaxSpeedConstrained.calculate_real_position_delta_1D(start_position, max_velocity, target_position,
    #                                                                                           time_delta)
    #
    #     self.assertEqual(real_position_delta, must_position_delta)
    #
    # @parameterized.expand([
    #     [[0],     120,    [100],    [100]],
    #     [[0],     80,     [100],    [80]],
    #     [[200],   120,    [100],    [100]],
    #     [[200],   80,     [100],    [120]],
    # ])
    # def test_calculate_real_position_1D(self, start_position, max_velocity, target_position, must_position):
    #     start_position = np.array(start_position)
    #     target_position = np.array(target_position)
    #     must_position = np.array(must_position)
    #     start_time = 0
    #     time_delta = 1
    #     sut = NewPositionMaxSpeedConstrained(start_time, start_position, max_velocity)
    #     real_position = sut._calculate_new_position(target_position, time_delta)
    #
    #     self.assertEqual(real_position, must_position)
    #
    # @parameterized.expand([
    #     [(0, 0),        120,        (100, 100),         (84, 84)],
    #     [(0, 0),        120,        (100, 0),           (100, 0)],
    #     [(0, 0),        120,        (0, 100),           (0, 100)],
    #     [(0, 0),        80,         (100, 100),         (56, 56)],
    #     # [(200, 400),    120,        (100, 100),         (100, 280)],
    #     # [(200, 400),    80,         (100, 100),         (120, 320)],
    # ])
    # def test_calculate_real_position_2D(self, start_position, max_velocity, target_position, must_position):
    #     start_position = np.array(start_position)
    #     target_position = np.array(target_position)
    #     must_position = np.array(must_position)
    #
    #     start_time = 0
    #     time_delta = 1
    #     sut = NewPositionMaxSpeedConstrained(start_time, start_position, max_velocity)
    #
    #     real_position = sut._calculate_new_position(target_position, time_delta)
    #     print(real_position)
    #
    #     print(must_position)
    #     np.testing.assert_array_equal(real_position, must_position)
