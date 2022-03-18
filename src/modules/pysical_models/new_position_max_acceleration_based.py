import math

import numpy as np

class NewPositionMaxAcceleartionBased:
    def __init__(self, time, position, v_max, a_max):
        self.current_time = time
        # #self.position_bounds = (0, 1920, 0, 1080)
        # self.acceleration_max = acceleration_max * 1.
        self.position = position * 1.

    def get_position(self):
        return tuple(self.position)

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
        max_velocity_for_time_delta = self.acceleration_max * time_delta
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


if __name__ == '__main__':
    a = -1
    v_0 = 3
    s_0 = 0

    v_t = 0
    s_t = 4.5


