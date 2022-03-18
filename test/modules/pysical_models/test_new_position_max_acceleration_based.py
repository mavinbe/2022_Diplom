import numpy as np
import unittest
from parameterized import parameterized

from modules.pysical_models.acceleration_movement_model import AccelerationMovementModel
from modules.pysical_models.new_position_max_acceleration_based import NewPositionMaxAcceleartionBased
from visualization.plot_movement import plot_movement


class TestNewPositionMaxAcceleartionBased(unittest.TestCase):

    @parameterized.expand([
        [{
            "title": "perfekt match",
            "a_0": -1,
            "v_0": 3,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4.5,
            "change_a": False
        }],
        [{
            "title": "to slow",
            "a_0": -1,
            "v_0": 2,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "change_a": True
        }],
        [{
            "title": "to fast",
            "a_0": -1,
            "v_0": 3,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "change_a": False
        }],
        [{
            "title": "to fast 2",
            "a_0": -1,
            "v_0": 4,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "change_a": False
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




if __name__ == '__main__':
    unittest.main()
