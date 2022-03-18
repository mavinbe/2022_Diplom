import numpy as np
import matplotlib.pyplot as plt

def plot_movement(sut, a_0, s_0, v_0, v_target, t_where_v_is_target, s_target, t_target, title="?"):

    res = 10
    movement = sut
    t = np.linspace(0, 10, 11 * res)
    # make data
    a = [a_0 for i in t]
    v = [movement.v(t=i, a=a_0, v_0=v_0) for i in t]
    s = [movement.s(t=i, a=a_0, v_0=v_0, s_0=s_0) for i in t]
    # plot
    fig, ax = plt.subplots(figsize=(25, 25))
    fig.suptitle(title, fontsize=64)
    #ax.plot(t, a, 'm', linewidth=2.0)
    ax.plot(t, v, 'g', linewidth=2.0)
    ax.plot(t, s, 'b', linewidth=2.0)
    ax.set(xlim=(0, 8), xticks=np.arange(0, 8),
           ylim=(-2, 8), yticks=np.arange(-2, 8))
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.axhline(y=s_target, color='b', linestyle=':')
    if t_target:
        ax.axvline(x=t_target, color='b', linestyle=':')
    ax.axhline(y=v_target, color='g', linestyle=':')
    if t_where_v_is_target:
        ax.axvline(x=t_where_v_is_target, color='g', linestyle=':')
    plt.show()