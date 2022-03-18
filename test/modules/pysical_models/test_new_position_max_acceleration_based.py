import numpy as np
import unittest
from parameterized import parameterized

from modules.pysical_models.acceleration_movement_model import AccelerationMovementModel
from modules.pysical_models.new_position_max_acceleration_based import NewPositionMaxAcceleartionBased
from visualization.plot_movement import plot_movement


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

        sut = NewPositionMaxAcceleartionBased(s_target=paras["s_target"], v_target=paras["v_target"], v_max=120, a_max=1)
        plot_movement(AccelerationMovementModel, a_0, s_0, v_0, v_target, None, s_target, None, paras["title"])
        self.assertEqual(sut.to_change_a(a_0=a_0, v_0=v_0, s_0=s_0, v_target=v_target, s_target=s_target), paras["change_a"])


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
