import sys
sys.path.insert(0, '/home/mavinbe/2021_Diplom/2021_Diplom_Lab/Kalman-and-Bayesian-Filters-in-Python')
from math import sqrt

from filterpy.kalman import FixedLagSmoother, KalmanFilter
# from filterpy.common import Q_discrete_white_noise
import numpy.random as random
import numpy as np

import matplotlib.pyplot as plt


from kf_book.book_plots import plot_measurements
from kf_book.book_plots import plot_filter

random.seed(2)


class PosSensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = np.asarray(vel)
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        self.count = 0

    def read(self):
        self.count += 1
        # if self.count == 40:
        #     self.vel[1] *= 4
        # print(self.vel)
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]


        sensor = [self.pos[0] + random.randn() * self.noise_std,
                self.pos[1] + random.randn() * self.noise_std]
        nom = [self.pos[0], self.pos[1]]
        return sensor, nom

# Sensor Noise
R_sensor = 20
# Measurement Noise
R_std = sqrt(1)
# Process Noise
Q_std = 0.00001

fls = FixedLagSmoother(dim_x=4, dim_z=2, N=8)
dt = 1.0  # time step

fls.x = np.array([[0, 0, 0, 0]]).T
fls.F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])

# fls.H = np.array([[1 / 0.3048, 0, 0, 0],
#                   [0, 0, 1 / 0.3048, 0]])
fls.H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

fls.P = np.eye(4) * 500.
fls.R = np.eye(2) * R_std ** 2
fls.Q *= Q_std

# simulate robot movement
N = 100
sensor = PosSensor((0, 0), (1, 1), noise_std=R_sensor)
# zs = np.array([])
# for _ in range(N):
#     c_zs, c_nom = sensor.read()
#     zs = np.array([zs])
#     nom = np.array([nom])
# print(zs)
# print(nom)

s_read = np.array([sensor.read() for _ in range(N)])
zs = s_read[:, 0, :]
nom = s_read[:, 1, :]

for z in zs:
    fls.smooth(z)

# kf_x, _, _, _ = kf.batch_filter(zs)
x_smooth = np.array(fls.xSmooth)[:, 0]

#fls_res = abs(x_smooth - nom)
#print(f'standard deviation fixed-lag: {np.mean(fls_res):.3f}')
# kf_res = abs(kf_x[:, 0] - nom)


#zs *= .3048
plot_measurements(zs[:, 0], zs[:, 1], None, 'r')
plot_filter(x_smooth[:, 0], x_smooth[:, 1], None, 'g', "fls")
plot_filter(nom[:, 0], nom[:, 1], None, 'b', "nom")

plt.legend(loc=1)
plt.show()
plt.xlim(0, 200)
plt.ylim(0, 200)

# print(f'standard deviation kalman: {np.mean(kf_res):.3f}')
