import math

import numpy as np

from modules.pysical_models.acceleration_movement_model import AccelerationMovementModel as Model


class NewPositionMaxAcceleartionBased:
    def __init__(self, s_target, v_target, v_max, a_max):
        self.s_target = s_target * 1.
        self.v_target = v_target * 1.
        self.v_max = v_max * 1.
        self.a_max = a_max * 1.

        # #self.position_bounds = (0, 1920, 0, 1080)

    # immutable
    def calculate_model_state(self, time_delta, s_0, v_0, a_0):
        s_0 = np.asarray(s_0)
        v_0 = np.asarray(v_0)
        a_0 = np.asarray(a_0)
        a = self.determ_a()

        return s, v, a

    def to_change_a(self, a_0, v_0, s_0, v_target, s_target):

        t_where_v_is_target = Model.calculate_t_given_v(v=v_target, a=a_0, v_0=v_0)


        t_next_where_s_is_target = Model.calculate_t_given_s_nearest_ahead(s=s_target, a=a_0, v_0=v_0, s_0=s_0)



        a_is = None
        if t_next_where_s_is_target == t_where_v_is_target:
            return False
        elif t_next_where_s_is_target is None:
            return True
        elif t_next_where_s_is_target < t_where_v_is_target:
            return False



    def determ_a(self):
        pass


    # mutable
    def move_to_target(self, velocity_target, position_target, new_time):
        target = np.asarray(position_target)
        time_delta = self._calculate_time_delta(new_time)
        new_position = self._calculate_new_position(target, time_delta)
        self.current_time = new_time
        self.position = new_position

    # immutable
    def _calculate_new_position(self, target, time_delta):
        # print(self.position.shape)
        # print(self.position)
        # print(target.shape)
        # print(target)
        if self.position.shape != target.shape:
            raise RuntimeError("self.position.shape != target.shape")
        max_velocity_for_time_delta = self.a_max * time_delta
        velocity = NewPositionMaxAcceleartionBased.calculate_velocity_for_dimensions(self.position, max_velocity_for_time_delta, target)

        return self.position + velocity

    # immutable
    def _calculate_time_delta(self, new_time):
        time_delta = new_time - self.current_time
        return time_delta

    # pure
    @staticmethod
    def calculate_velocity_for_dimensions(start_vector, max_velocity, target_vector):
        delta_vector = target_vector - start_vector
        delta_vector_magnitude = np.linalg.norm(delta_vector)
        if delta_vector_magnitude == 0:
            delta_vector_unit = delta_vector * 0
        else:
            delta_vector_unit = delta_vector / delta_vector_magnitude

        velocity = delta_vector_unit * min(delta_vector_magnitude, max_velocity)

        return velocity



