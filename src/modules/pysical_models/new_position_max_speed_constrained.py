import numpy as np

class NewPositionMaxSpeedConstrained:
    def __init__(self, time, position, max_velocity):
        self.current_time = time
        #self.position_bounds = (0, 1920, 0, 1080)
        self.max_velocity = max_velocity * 1.
        self.position = position * 1.

    def get_position(self):
        return tuple(self.position)

    # mutable
    def move_to_target(self, target, new_time):
        target = np.asarray(target)
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
            raise RuntimeError("self.position.shape != target.shape "+ str(self.position.shape)+" "+str(target.shape))
        velocity_for_time_delta = self._calculate_velocity_based_on_distance(time_delta, self.position, target)

        #velocity_for_time_delta = self.max_velocity * time_delta
        print(velocity_for_time_delta)
        velocity = NewPositionMaxSpeedConstrained.calculate_velocity_for_dimensions(self.position, velocity_for_time_delta, target)

        return self.position + velocity

    def _calculate_velocity_based_on_distance(self, time_delta, start_vector,  target_vector):
        delta_vector = target_vector - start_vector
        delta_vector_magnitude = np.linalg.norm(delta_vector)
        velocity = delta_vector_magnitude * 4
        velocity = min(velocity, self.max_velocity)

        velocity_for_time_delta = velocity * time_delta
        return velocity_for_time_delta

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
