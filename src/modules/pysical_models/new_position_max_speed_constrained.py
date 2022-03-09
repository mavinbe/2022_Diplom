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
        self.max_velocity = max_velocity
        #self.max_acceleration = 300

        self.position = position
        ##self.acceleration = (0, 0)

    def get_position(self):
        return self.position

    def move_to_target(self, target, new_time):
        time_delta = self._calculate_time_delta(new_time)
        new_position = self.calculate_new_position(target, time_delta)
        self.position = new_position

    def calculate_new_position(self, target, time_delta):
        if self.position.shape != target.shape:
            raise RuntimeError("self.position.shape != target.shape")
        new_position = np.asarray(self.position)
        for idx, dimension in enumerate(target):
            print(idx)
            print(dimension)
            new_position[idx] = self.position[idx] + self.calculate_real_position_delta_1D(self.position[idx], self.max_velocity, target[idx], time_delta)
        return new_position

    def _calculate_time_delta(self, new_time):
        return new_time - self.current_time

    @staticmethod
    def calculate_real_position_delta_1D(position, max_velocity, target, time_delta):
        total_position_delta = target - position
        direction_norm = abs(total_position_delta) / total_position_delta
        calculated_position_delta = NewPositionMaxSpeedConstrained.calculate_position_delta(max_velocity * direction_norm, time_delta)

        real_position_delta = None
        if total_position_delta == 0:
            real_position_delta = 0
        elif total_position_delta > 0:
            real_position_delta = min(total_position_delta, calculated_position_delta)
        elif total_position_delta < 0:
            real_position_delta = max(total_position_delta, calculated_position_delta)
        return real_position_delta

    @staticmethod
    def calculate_position_delta(velocity, time_delta):
        return velocity * time_delta

