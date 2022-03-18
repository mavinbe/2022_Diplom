import numpy as np
import matplotlib.pyplot as plt

def plot_movement(t_list, a_list, s_list, v_list, v_target, t_where_v_is_target, s_target, t_target, title="?"):


    # plot
    fig, ax = plt.subplots(figsize=(25, 25))
    fig.suptitle(title, fontsize=64)
    #ax.plot(t, a_list, 'm', linewidth=2.0)
    ax.plot(t_list, v_list, 'g', linewidth=2.0)
    ax.plot(t_list, s_list, 'b', linewidth=2.0)
    ax.set(xlim=(0, 8), xticks=np.arange(0, 8),
           ylim=(-8, 8), yticks=np.arange(-8, 8))
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.axhline(y=s_target, color='b', linestyle=':')
    if t_target:
        ax.axvline(x=t_target, color='b', linestyle=':')
    ax.axhline(y=v_target, color='g', linestyle=':')
    if t_where_v_is_target:
        ax.axvline(x=t_where_v_is_target, color='g', linestyle=':')
    plt.show()