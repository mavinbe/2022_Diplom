from interval import interval, inf, imath
import numpy as np

class MovementPredictionModel:
    def __init__(self, a_max, v_start, s_start, intervals):
        self._a_max = a_max
        self._v_start = v_start
        self._s_start = s_start
        self._interval_up, self._interval_neutral, self._interval_down = [interval(i) for i in intervals]

    def a_up(self):
        return abs(self._a_max)

    def a_neutral(self):
        return 0

    def a_down(self):
        return -1 * self.a_up()

    def a_for_plot(self, t):
        self.check_if_t_is_in_interval_or_raise_excepion(t)
        if t in self._interval_up:
            if t in self._interval_up.extrema:
                return np.nan
        elif t in self._interval_neutral:
            if t in self._interval_neutral.extrema:
                return np.nan
        elif t in self._interval_down:
            if t in self._interval_down.extrema:
                return np.nan
        return self.a(t)

    def a(self, t):
        self.check_if_t_is_in_interval_or_raise_excepion(t)
        if t in self._interval_up:
            return self.a_up()
        elif t in self._interval_neutral:
            return self.a_neutral()
        elif t in self._interval_down:
            return self.a_down()

    def v(self, t):
        self.check_if_t_is_in_interval_or_raise_excepion(t)
        min = None
        if t in self._interval_up:
            min = self._interval_up.extrema[0][0]
            return self.a(t) * (t - min) + self._v_start
        elif t in self._interval_neutral:
            min = self._interval_neutral.extrema[0][0]
            return self.a(t) * (t - min) + self.v(min)
        elif t in self._interval_down:
            min = self._interval_down.extrema[0][0]
            return self.a(t) * (t - min) + self.v(min)

    def s(self, t):
        self.check_if_t_is_in_interval_or_raise_excepion(t)
        min = None
        if t in self._interval_up:
            min = self._interval_up.extrema[0][0]
            return 0.5 * self.a(t) * (t - min) **2 + self._v_start * (t - min) + self._s_start
        elif t in self._interval_neutral:
            min = self._interval_neutral.extrema[0][0]
            return 0.5 * self.a(t) * (t - min) **2 + self.v(min) * (t - min) + self.s(min)
        elif t in self._interval_down:
            min = self._interval_down.extrema[0][0]
            return 0.5 * self.a(t) * (t - min) **2 + self.v(min) * (t - min) + self.s(min)




    def check_if_t_is_in_interval_or_raise_excepion(self, t):
        if t in self._interval_up or t in self._interval_neutral or t in self._interval_down:
            return
        raise RuntimeError("t ist not in any interval")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    res = 100

    # movement = MovementPredictionModel(1, 0, 0, ([0, 2], [2, 4], [4, 6]))
    # t = np.linspace(0, 2, 3*res), np.linspace(2, 4, 3*res), np.linspace(4, 6, 3*res)

    # movement = MovementPredictionModel(1, 1, 0, ([0, 1], [1, 3], [3, 5]))
    # t = np.linspace(0, 1, 2*100), np.linspace(1, 3, 3*res), np.linspace(3, 5, 3*res)

    # movement = MovementPredictionModel(1, 3, 0, ([0, 0], [0, 0], [0, 5]))
    # t = np.linspace(0, 1, 2 * 100), np.linspace(1, 3, 3 * res), np.linspace(3, 5, 3 * res)

    movement = MovementPredictionModel(1, 2, 0, ([0, 0.55], [0, 0], [0.55, 5]))
    t = np.linspace(0, 1, 2 * 100), np.linspace(1, 3, 3 * res), np.linspace(3, 5, 3 * res)

    # make data

    a = [[movement.a(i) for i in t_i] for t_i in t]
    v = [[movement.v(i) for i in t_i] for t_i in t]
    s = [[movement.s(i) for i in t_i] for t_i in t]

    # plot
    fig, ax = plt.subplots(figsize=(25,25))

    for i in range(0, 3):
        ax.plot(t[i], a[i], 'm', linewidth=2.0)
        ax.plot(t[i], v[i], 'g', linewidth=2.0)
        ax.plot(t[i], s[i], 'b', linewidth=2.0)

    #ax.plot(t, s, linewidth=2.0)

    ax.set(xlim=(0, 8), xticks=np.arange(0, 8),
           ylim=(-8, 8), yticks=np.arange(-8, 8))
    ax.axhline(y=0, color='k', linestyle=':')
    ax.axvline(x=0, color='k')
    ax.axhline(y=0, color='g', linestyle=':')
    ax.axhline(y=4.5, color='b', linestyle=':')
    print(f'v {v[-1][-1]}')
    print(f's {s[-1][-1]}')

    plt.show()