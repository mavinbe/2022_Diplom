import math

import numpy as np


class AccelerationMovementModel:

    @classmethod
    def state(cls, t, a, v_0, s_0):
        return a, cls.v(t, a, v_0), cls.s(t, a, v_0, s_0)

    @classmethod
    def v(cls, t, a, v_0):
        return a * t + v_0

    @classmethod
    def s(cls, t, a, v_0, s_0):
        return a / 2 * t ** 2 + v_0 * t + s_0

    @classmethod
    def calculate_t_given_v(cls, v, a, v_0):
        return (v - v_0) / a

    @classmethod
    def calculate_t_given_s(cls, s, a, v_0, s_0):
        front = - (v_0/a)
        in_sqrt =  (v_0/a)**2 - 2*s_0/a + 2*s/a
        back = None
        if in_sqrt < 0:
            return None
        try:
            back = math.sqrt(in_sqrt)
        except:
            print(in_sqrt)
        return (front - back, front + back)

    @classmethod
    def calculate_t_given_s_nearest_ahead(cls, s, a, v_0, s_0):
        t = cls.calculate_t_given_s(s, a, v_0, s_0)
        if t is None:
            return None
        if 0 > t[0] and 0 > t[1]:
            return None
        elif 0 <= t[0] and 0 <= t[1]:
            return t[0]
        else:
            return t[1]


