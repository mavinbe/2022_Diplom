from math import sqrt

import numpy as np


def test_dot():
    unit_vector_x = np.array([1., 0.])
    unit_vector_y = np.array([0, 1.])
    print(f'unit_vector_x: {unit_vector_x}')
    print(f'unit_vector_y: {unit_vector_y}')

    target_vector = np.array([2., 1.])
    print(f'target_vector: {target_vector}')
    target_vector_unit = target_vector / np.linalg.norm(target_vector)
    magnitude = np.linalg.norm(target_vector) * 2
    print(f'magnitude: {magnitude}')
    real_vector = target_vector_unit * magnitude

    print(f'real_vector: {real_vector}')

    x = np.dot(real_vector, unit_vector_x)
    y = np.dot(real_vector, unit_vector_y)

    print(f'x: {x}')
    print(f'y: {y}')

print(np.eye(3))
print(np.array([2,3]).shape[0])

#test_dot()
