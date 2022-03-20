import numpy as np
import unittest
from parameterized import parameterized

from modules.pysical_models.acceleration_movement_model import AccelerationMovementModel
from modules.pysical_models.new_position_max_acceleration_based import NewPositionMaxAcceleartionBased
from visualization.plot_movement import plot_movement, plot_movement_2D


class TestNewPositionMaxAcceleartionBased(unittest.TestCase):

    @parameterized.expand([

        [{
            "title": "to slow",
            "a_0": -1,
            "v_0": 2,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4.5,
            "v_direction": -1,
            "to_change_a": True,
            "determ_a_without_v_constrains": 1,
            "calculate_a": 1,
        }],
        [{
            "title": "perfekt match",
            "a_0": -1,
            "v_0": 3,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4.5,
            "v_direction": -1,
            "to_change_a": False,
            "determ_a_without_v_constrains": -1,
            "calculate_a": -1,
        }],
        [{
            "title": "to fast",
            "a_0": -1,
            "v_0": 3,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "v_direction": -1,
            "to_change_a": False,
            "determ_a_without_v_constrains": -1,
            "calculate_a": -1,
        }],
        [{
            "title": "to fast 2",
            "a_0": -1,
            "v_0": 4,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "v_direction": -1,
            "to_change_a": False,
            "determ_a_without_v_constrains": -1,
            "calculate_a": -1,
        }],
    ])
    def test_to_change_a(self, paras):
        a_0 = paras["a_0"]
        v_0 = paras["v_0"]
        s_0 = paras["s_0"]
        v_target = paras["v_target"]
        s_target = paras["s_target"]

        sut = NewPositionMaxAcceleartionBased(s_target=paras["s_target"], v_target=paras["v_target"], v_max=5, a_max=1)

        a_list, s_list, t, v_list = self.generate_plot_data(a_0, s_0, v_0, 10)

        #plot_movement(t, a_list, s_list, v_list, v_target, None, s_target, None, paras["title"])

        self.assertEqual(sut.v_direction(v_0=v_0, v_target=v_target), paras["v_direction"])
        self.assertEqual(sut.to_change_a(a_0=a_0, v_0=v_0, s_0=s_0, v_target=v_target, s_target=s_target), paras["to_change_a"])
        self.assertEqual(sut.determ_a_without_v_constrains(v_0=v_0, s_0=s_0, v_target=v_target, s_target=s_target),
                         paras["determ_a_without_v_constrains"])
        a = sut.calculate_a(1, v_0, s_0)
        self.assertEqual(a,
                         paras["calculate_a"])


    @parameterized.expand([

        [{
            "title": "to slow",
            "a_0": -1,
            "v_0": 2,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4.5,
            "v_direction": -1,
            "to_change_a": True,
            "determ_a_without_v_constrains": 1,
            "calculate_a": 1,
        }],
        [{
            "title": "perfekt match",
            "a_0": -1,
            "v_0": 3,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4.5,
            "v_direction": -1,
            "to_change_a": False,
            "determ_a_without_v_constrains": -1,
            "calculate_a": -1,
        }],
        [{
            "title": "to fast",
            "a_0": -1,
            "v_0": 3,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "v_direction": -1,
            "to_change_a": False,
            "determ_a_without_v_constrains": -1,
            "calculate_a": -1,
        }],
        [{
            "title": "to fast 2",
            "a_0": -1,
            "v_0": 4,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "v_direction": -1,
            "to_change_a": False,
            "determ_a_without_v_constrains": -1,
            "calculate_a": -1,
        }],
    ])
    def test_visualize_optimal_solution(self, paras):
        a_0 = paras["a_0"]
        v_0 = paras["v_0"]
        s_0 = paras["s_0"]
        v_target = paras["v_target"]
        s_target = paras["s_target"]

        sut = NewPositionMaxAcceleartionBased(s_target=paras["s_target"], v_target=paras["v_target"], v_max=5, a_max=1)

        res = 10
        a, v, s = a_0, v_0, s_0

        a_list, s_list, v_list = [], [], []
        t = np.linspace(0, 10, 11 * res)
        a_last, v_last, s_last = AccelerationMovementModel.state(0, 1, v, s)

        t_last = 0
        for _t in t:

            t_delta = _t - t_last
            t_last = _t
            a_last = sut.calculate_a(time_delta=t_delta, v_0=v_last, s_0=s_last)

            _, v_last, s_last = AccelerationMovementModel.state(t_delta, a_last, v_last, s_last)
            if abs(s_target - s_last) < 0.01 and abs(v_target - v_last) < 0.05:
                a_last, v_last, s_last = AccelerationMovementModel.state(t_delta, 0, v_target, s_target)

            a_list.append(a_last)
            v_list.append(v_last)
            s_list.append(s_last)

        #plot_movement(t, a_list, s_list, v_list, v_target, None, s_target, None, f'optimal_solution - {paras["title"]}')

    @parameterized.expand([

        [{
            "title": "to slow",
            "a_0": [-1,0],
            "v_0": [2,0],
            "s_0": [0,0],
            "v_target": [0,0],
            "s_target": [4.5,0],
            "v_direction": [-1,0],
            "to_change_a": [True, False],
            "determ_a_without_v_constrains": [1,0],
            "calculate_a": [1,0],
        }],
        # [{
        #     "title": "perfekt match",
        #     "a_0": -1,
        #     "v_0": 3,
        #     "s_0": 0,
        #     "v_target": 0,
        #     "s_target": 4.5,
        #     "v_direction": -1,
        #     "to_change_a": False,
        #     "determ_a_without_v_constrains": -1,
        #     "calculate_a": -1,
        # }],
        # [{
        #     "title": "to fast",
        #     "a_0": -1,
        #     "v_0": 3,
        #     "s_0": 0,
        #     "v_target": 0,
        #     "s_target": 4,
        #     "v_direction": -1,
        #     "to_change_a": False,
        #     "determ_a_without_v_constrains": -1,
        #     "calculate_a": -1,
        # }],
        # [{
        #     "title": "to fast 2",
        #     "a_0": -1,
        #     "v_0": 4,
        #     "s_0": 0,
        #     "v_target": 0,
        #     "s_target": 4,
        #     "v_direction": -1,
        #     "to_change_a": False,
        #     "determ_a_without_v_constrains": -1,
        #     "calculate_a": -1,
        # }],
    ])
    def test_visualize_optimal_solution_2D(self, paras):
        a_0 = paras["a_0"]
        v_0 = paras["v_0"]
        s_0 = paras["s_0"]
        v_target = paras["v_target"]
        s_target = paras["s_target"]

        sut = NewPositionMaxAcceleartionBased(s_target=paras["s_target"], v_target=paras["v_target"], v_max=5,
                                              a_max=1)

        res = 10
        a, v, s = a_0, v_0, s_0

        a_list, s_list, v_list = [], [], []
        t = np.linspace(0, 10, 11 * res)
        a_last, v_last, s_last = AccelerationMovementModel.state(0, 1, v, s)

        t_last = 0
        for _t in t:

            t_delta = _t - t_last
            t_last = _t
            a_last = sut.calculate_a(time_delta=t_delta, v_0=v_last, s_0=s_last)

            _, v_last, s_last = AccelerationMovementModel.state(t_delta, a_last, v_last, s_last)
            if abs(s_target - s_last) < 0.01 and abs(v_target - v_last) < 0.05:
                a_last, v_last, s_last = AccelerationMovementModel.state(t_delta, 0, v_target, s_target)

            a_list.append(a_last)
            v_list.append(v_last)
            s_list.append(s_last)


        plot_movement_2D(t, [a_list, a_list], [s_list, s_list], [v_list, v_list], [v_target, v_target],
                         [None, None], [s_target, s_target],
                         [None, None],
                         f'optimal_solution - {paras["title"]}')


    def generate_plot_data(self, a_0, s_0, v_0, res):
        movement = AccelerationMovementModel
        t = np.linspace(0, 10, 11 * res)
        # make data
        a_list = [a_0 for i in t]
        v_list = [movement.v(t=i, a=a_0, v_0=v_0) for i in t]
        s_list = [movement.s(t=i, a=a_0, v_0=v_0, s_0=s_0) for i in t]
        return a_list, s_list, t, v_list


if __name__ == '__main__':
    unittest.main()
