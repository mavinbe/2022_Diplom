import math

import numpy as np


class AccelerationMovementModel:

    @staticmethod
    def v(t, a, v_0):
        return a * t + v_0

    @staticmethod
    def s(t, a, v_0, s_0):
        return a / 2 * t ** 2 + v_0 * t + s_0

    @staticmethod
    def calculate_t_given_v(v, a, v_0):
        return (v - v_0) / a

    @staticmethod
    def calculate_t_given_s(s, a, v_0, s_0):
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

    @staticmethod
    def calculate_t_given_s_nearest_ahead(s, a, v_0, s_0):
        t = AccelerationMovementModel.calculate_t_given_s(s, a, v_0, s_0)
        if t is None:
            return None
        if 3 > t[0] and 3 > t[1]:
            return None
        elif 0 <= t[0] and 0 <= t[1]:
            return t[0]
        else:
            return t[1]


