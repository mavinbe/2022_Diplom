import numpy as np
from utils.general import check_imshow


class NewPositionMaxSpeedConstrained:
    def __init__(self, time, position, max_velocity, show_vid=False):
        # Check if environment supports image displays
        if show_vid:
            show_vid = check_imshow()
        self.show_vid = show_vid

        self.current_time = time

        self.position_bounds = (0, 1920, 0, 1080)
        self.max_velocity = max_velocity * 1.
        #self.max_acceleration = 300

        self.position = position * 1.
        ##self.acceleration = (0, 0)

    def get_position(self):
        return tuple(self.position)

    def move_to_target(self, target, new_time):
        target = np.asarray(target)
        time_delta = self._calculate_time_delta(new_time)
        new_position = self.calculate_new_position(target, time_delta)
        self.position = new_position

    def calculate_new_position(self, target, time_delta):
        if self.position.shape != target.shape:
            raise RuntimeError("self.position.shape != target.shape")

        velocity = NewPositionMaxSpeedConstrained.calculate_velocity_for_dimensions(self.position, self.max_velocity, target)

        return self.position + velocity

    def _calculate_time_delta(self, new_time):
        return new_time - self.current_time

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
