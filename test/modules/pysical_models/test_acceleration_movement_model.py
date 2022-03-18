import numpy as np
import unittest
from parameterized import parameterized

import matplotlib.pyplot as plt

from modules.pysical_models.acceleration_movement_model import AccelerationMovementModel


def plot_movement(sut, a_0, s_0, v_0, v_target, t_where_v_is_target, s_target, t_target, title="?"):

    res = 10
    movement = sut
    t = np.linspace(0, 10, 11 * res)
    # make data
    a = [a_0 for i in t]
    v = [movement.v(t=i, a=a_0, v_0=v_0) for i in t]
    s = [movement.s(t=i, a=a_0, v_0=v_0, s_0=s_0) for i in t]
    # plot
    fig, ax = plt.subplots(figsize=(25, 25))
    fig.suptitle(title, fontsize=64)
    #ax.plot(t, a, 'm', linewidth=2.0)
    ax.plot(t, v, 'g', linewidth=2.0)
    ax.plot(t, s, 'b', linewidth=2.0)
    ax.set(xlim=(0, 8), xticks=np.arange(0, 8),
           ylim=(-2, 8), yticks=np.arange(-2, 8))
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.axhline(y=s_target, color='b', linestyle=':')
    if t_target:
        ax.axvline(x=t_target, color='b', linestyle=':')
    ax.axhline(y=v_target, color='g', linestyle=':')
    ax.axvline(x=t_where_v_is_target, color='g', linestyle=':')
    plt.show()


class TestAccelerationMovementModel(unittest.TestCase):

    @parameterized.expand([
        [{
            "title": "perfekt match",
            "a_0": -1,
            "v_0": 3,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4.5,
            "t_where_v_is_target": 3,
            "s_on_t_where_v_is_target": 4.5,
            "t_next_where_s_is_target": 3,
            "a_is": "perfekt"
        }],
        [{
            "title": "to fast",
            "a_0": -1,
            "v_0": 3,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "t_where_v_is_target": 3,
            "s_on_t_where_v_is_target": 4.5,
            "t_next_where_s_is_target": 2,
            "a_is": "to high"
        }],
        [{
            "title": "to slow",
            "a_0": -1,
            "v_0": 2,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "t_where_v_is_target": 2,
            "s_on_t_where_v_is_target": 2,
            "t_next_where_s_is_target": None,
            "a_is": "to low"
        }],
        [{
            "title": "to fast 2",
            "a_0": -1,
            "v_0": 4,
            "s_0": 0,
            "v_target": 0,
            "s_target": 4,
            "t_where_v_is_target": 4,
            "s_on_t_where_v_is_target": 8,
            "t_next_where_s_is_target": 1.1715728752538097,
            "a_is": "to high"
        }],
    ])
    def test_t_from_s(self, sdf):
        a_0 = sdf["a_0"]
        v_0 = sdf["v_0"]
        s_0 = sdf["s_0"]
        v_target = sdf["v_target"]
        s_target = sdf["s_target"]

        sut = AccelerationMovementModel

        t_where_v_is_target = sut.calculate_t_given_v(v=v_target, a=a_0, v_0=v_0)
        self.assertEqual(t_where_v_is_target, sdf["t_where_v_is_target"])
        self.assertEqual(sut.s(t=t_where_v_is_target, a=a_0, v_0=v_0, s_0=s_0), sdf["s_on_t_where_v_is_target"])

        t_next_where_s_is_target = sut.calculate_t_given_s_nearest_ahead(s=s_target, a=a_0, v_0=v_0, s_0=s_0)
        self.assertEqual(t_next_where_s_is_target, sdf["t_next_where_s_is_target"])

        plot_movement(sut, a_0, s_0, v_0, v_target, t_where_v_is_target, s_target, t_next_where_s_is_target,
                      sdf["title"])
        a_is = None
        if t_next_where_s_is_target == t_where_v_is_target:
            a_is = "perfekt"
        elif t_next_where_s_is_target is None:
            a_is = "to low"
        elif t_next_where_s_is_target < t_where_v_is_target:
            a_is = "to high"
        self.assertIsNotNone(a_is)
        self.assertEqual(a_is, sdf["a_is"])

    def test_v(self):
        sut = AccelerationMovementModel

        self.assertEqual(sut.v(t=0, a=1, v_0=0), 0)
        self.assertEqual(sut.v(t=1, a=1, v_0=0), 1)
        self.assertEqual(sut.v(t=2, a=1, v_0=0), 2)
        self.assertEqual(sut.v(t=3, a=1, v_0=0), 3)

        self.assertEqual(sut.v(t=0, a=1, v_0=100), 100)
        self.assertEqual(sut.v(t=1, a=1, v_0=100), 101)
        self.assertEqual(sut.v(t=2, a=1, v_0=100), 102)

        self.assertEqual(sut.v(t=0, a=-1, v_0=0), 0)
        self.assertEqual(sut.v(t=1, a=-1, v_0=0), -1)

        self.assertEqual(sut.v(t=0, a=2, v_0=0), 0)
        self.assertEqual(sut.v(t=1, a=2, v_0=0), 2)
        self.assertEqual(sut.v(t=2, a=2, v_0=0), 4)

    def test_calculate_t_given_v(self):
        sut = AccelerationMovementModel

        self.assertEqual(sut.calculate_t_given_v(v=0, a=-1, v_0=3), 3)
        self.assertEqual(sut.s(t=3, a=-1, v_0=3, s_0=0), 4.5)

        self.assertEqual(sut.calculate_t_given_v(v=1, a=-1, v_0=3), 2)
        self.assertEqual(sut.s(t=2, a=-1, v_0=3, s_0=0), 4)


    def test_s(self):
        sut = AccelerationMovementModel

        self.assertEqual(sut.s(t=0, a=1, v_0=0, s_0=0), 0)
        self.assertEqual(sut.s(t=1, a=1, v_0=0, s_0=0), 0.5)
        self.assertEqual(sut.s(t=2, a=1, v_0=0, s_0=0), 2)
        self.assertEqual(sut.s(t=3, a=1, v_0=0, s_0=0), 4.5)
        self.assertEqual(sut.s(t=4, a=1, v_0=0, s_0=0), 8)

        self.assertEqual(sut.s(t=0, a=1, v_0=0, s_0=10), 10)
        self.assertEqual(sut.s(t=4, a=1, v_0=0, s_0=10), 18)

        self.assertEqual(sut.s(t=0, a=1, v_0=10, s_0=0), 0)
        self.assertEqual(sut.s(t=1, a=1, v_0=10, s_0=0), 10.5)
        self.assertEqual(sut.s(t=2, a=1, v_0=10, s_0=0), 22)


if __name__ == '__main__':
    unittest.main()
