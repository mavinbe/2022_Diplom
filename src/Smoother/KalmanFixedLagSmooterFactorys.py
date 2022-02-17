from filterpy.kalman import FixedLagSmoother, KalmanFilter
import numpy as np

def ZeroOrderSmoother(_R_std, _Q_std):
    fls = FixedLagSmoother(dim_x=2, dim_z=2, N=8)
    dt = 1.0  # time step

    fls.x = np.array([[0, 0]]).T
    fls.F = np.array([[1, 0],
                      [0, 1]])

    fls.H = np.array([[1, 0],
                      [0, 1]])

    fls.P = np.eye(2) * 500.
    fls.R = np.eye(2) * _R_std ** 2
    fls.Q *= _Q_std
    return fls

def FirstOrderSmoother(_R_std, _Q_std):
    fls = FixedLagSmoother(dim_x=4, dim_z=2, N=8)
    dt = 1.0  # time step

    fls.x = np.array([[0, 0, 0, 0]]).T
    fls.F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]])

    fls.H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])

    fls.P = np.eye(4) * 500.
    fls.R = np.eye(2) * _R_std ** 2
    fls.Q *= _Q_std
    return fls

def SecondOrderSmoother(_R_std, _Q_std):
    fls = FixedLagSmoother(dim_x=6, dim_z=2, N=8)
    dt = 1.0  # time step

    fls.x = np.array([[0, 0, 0, 0, 0, 0]]).T
    fls.F = np.array([[1, dt, .5*dt*dt, 0, 0, 0],
                      [0, 1, dt, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, dt, .5*dt*dt],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]])

    fls.H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0]])

    fls.P = np.eye(6) * 500.
    fls.R = np.eye(2) * _R_std ** 2
    fls.Q *= _Q_std
    return fls

