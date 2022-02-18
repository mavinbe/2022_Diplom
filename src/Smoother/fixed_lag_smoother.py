
from numpy import dot
import numpy as np
from scipy.linalg import inv

from filterpy.kalman import FixedLagSmoother

class MyFixedLagSmoother(FixedLagSmoother):

    def smooth(self, z, u=None):
        """ Smooths the measurement using a fixed lag smoother.

        On return, self.xSmooth is populated with the N previous smoothed
        estimates,  where self.xSmooth[k] is the kth time step. self.x
        merely contains the current Kalman filter output of the most recent
        measurement, and is not smoothed at all (beyond the normal Kalman
        filter processing).

        self.xSmooth grows in length on each call. If you run this 1 million
        times, it will contain 1 million elements. Sure, we could minimize
        this, but then this would make the caller's code much more cumbersome.

        This also means that you cannot use this filter to track more than
        one data set; as data will be hopelessly intermingled. If you want
        to filter something else, create a new FixedLagSmoother object.

        Parameters
        ----------

        z : ndarray or scalar
            measurement to be smoothed


        u : ndarray, optional
            If provided, control input to the filter
        """

        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        B = self.B
        N = self.N

        k = self.count

        # predict step of normal Kalman filter
        x_pre = dot(F, x)
        if u is not None:
            x_pre += dot(B, u)

        P = dot(F, P).dot(F.T) + Q

        # update step of normal Kalman filter
        self.y = z - dot(H, x_pre)

        self.S = dot(H, P).dot(H.T) + R
        SI = inv(self.S)

        K = dot(P, H.T).dot(SI)

        x = x_pre + dot(K, self.y)

        I_KH = self._I - dot(K, H)
        P = dot(I_KH, P).dot(I_KH.T) + dot(K, R).dot(K.T)

        self.xSmooth.append(x_pre.copy())

        # compute invariants
        HTSI = dot(H.T, SI)
        F_LH = (F - dot(K, H)).T

        if k >= N:
            PS = P.copy()  # smoothed P for step i
            for i in range(N):
                K = dot(PS, HTSI)  # smoothed gain
                PS = dot(PS, F_LH)  # smoothed covariance

                si = k - i
                self.xSmooth[si] = self.xSmooth[si] + dot(K, self.y)
        else:
            # Some sources specify starting the fix lag smoother only
            # after N steps have passed, some don't. I am getting far
            # better results by starting only at step N.
            self.xSmooth[k] = x.copy()

        self.count += 1
        self.x = x
        self.P = P
