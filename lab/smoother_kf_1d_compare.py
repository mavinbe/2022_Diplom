#import sys
#sys.path.insert(0, '/home/mavinbe/2021_Diplom/2021_Diplom_Lab/Kalman-and-Bayesian-Filters-in-Python')

from filterpy.kalman import FixedLagSmoother, KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy.random as random
import numpy as np


import matplotlib.pyplot as plt
random.seed(2)

fls = FixedLagSmoother(dim_x=2, dim_z=1, N=8)

fls.x = np.array([0., .5])
fls.F = np.array([[1., 1.],
                  [0., 1.]])

fls.H = np.array([[1., 0.]])
fls.P *= 200
fls.R *= 5.
fls.Q *= 0.001

kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([0., .5])
kf.F = np.array([[1., 1.],
                 [0., 1.]])
kf.H = np.array([[1., 0.]])
kf.P *= 200
kf.R *= 5.
kf.Q = Q_discrete_white_noise(dim=2, dt=1., var=0.001)

#N = 4  # size of lag

nom = np.array([t / 2. for t in range(0, 40)])
zs = np.array([t + random.randn() * 5.1 for t in nom])

for z in zs:
    fls.smooth(z)

kf_x, _, _, _ = kf.batch_filter(zs)
x_smooth = np.array(fls.xSmooth)[:, 0]

fls_res = abs(x_smooth - nom)
kf_res = abs(kf_x[:, 0] - nom)

plt.plot(zs, 'o', alpha=0.5, marker='o', label='zs')
plt.plot(x_smooth, label='FLS')
plt.plot(kf_x[:, 0], label='KF', ls='--')
plt.legend(loc=4)
plt.show()

print(f'standard deviation fixed-lag: {np.mean(fls_res):.3f}')
print(f'standard deviation kalman: {np.mean(kf_res):.3f}')
