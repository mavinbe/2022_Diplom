import sys

sys.path.append('/home/mavinbe/2021_Diplom/2022_Diplom')
sys.path.append('/home/mavinbe/2021_Diplom/2021_Diplom_Lab/Kalman-and-Bayesian-Filters-in-Python')


from src.Smoother.KalmanFixedLagSmooterFactorys import SecondOrderSmoother, ZeroOrderSmoother, FirstOrderSmoother

from math import sqrt

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
        if self.count == 40:
            self.vel[1] = 8
        if self.count == 80:
            self.vel[1] = 1
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        sensor = [self.pos[0] + random.randn() * self.noise_std,
                  self.pos[1] + random.randn() * self.noise_std]
        nom = [self.pos[0], self.pos[1]]
        return sensor, nom


# Sensor Noise
R_sensor = 20
# Measurement Noise
R_std = sqrt(R_sensor)
# Process Noise
Q_std = 0.00001

fls = [
    (ZeroOrderSmoother(R_std, Q_std,    8), 'r'),
    (FirstOrderSmoother(R_std, Q_std,    8), 'g'),
    (SecondOrderSmoother(R_std, Q_std, 8), 'b')
]


# simulate robot movement
N = 500
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


def smooth_and_draw(zs, fls,  color):
    for z in zs:
        fls.smooth(z)
    x_smooth = np.array(fls.xSmooth)[:, 0]
    plot_filter(x_smooth[:, 0], x_smooth[:, 1], None, color, "fls")
    fls_res = abs(x_smooth - nom)
    print(f'standard deviation fixed-lag: {np.mean(fls_res):.3f}')


for current_fls in fls:
    (current_fls, color) = current_fls
    smooth_and_draw(zs, current_fls, color)


# kf_res = abs(kf_x[:, 0] - nom)


# zs *= .3048
plot_measurements(zs[:, 0], zs[:, 1], None, 'r')

plot_filter(nom[:, 0], nom[:, 1], None, 'b', "nom")

plt.legend(loc=1)
plt.show()
plt.xlim(0, 200)
plt.ylim(0, 200)

# print(f'standard deviation kalman: {np.mean(kf_res):.3f}')
