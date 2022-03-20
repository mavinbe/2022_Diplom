import numpy as np
import matplotlib.pyplot as plt

def plot_movement(t_list, a_list, s_list, v_list, v_target, t_where_v_is_target, s_target, t_target, title="?"):


    # plot
    fig, ax = plt.subplots(figsize=(25, 25))
    fig.suptitle(title, fontsize=64)
    #ax.plot(t, a_list, 'm', linewidth=2.0)
    ax.plot(t_list, v_list, 'g', linewidth=2.0)
    ax.plot(t_list, s_list, 'b', linewidth=2.0)
    ax.set(xlim=(0, 12), xticks=np.arange(0, 12),
           ylim=(-8, 9), yticks=np.arange(-8, 9))
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.axhline(y=s_target, color='b', linestyle=':')
    if t_target:
        ax.axvline(x=t_target, color='b', linestyle=':')
    ax.axhline(y=v_target, color='g', linestyle=':')
    if t_where_v_is_target:
        ax.axvline(x=t_where_v_is_target, color='g', linestyle=':')
    plt.show()

def plot_movement_2D(t_list, a_list_2D, s_list_2D, v_list_2D, v_target_2D, t_where_v_is_target_2D, s_target_2D, t_target_2D, title="?"):

    fig, ax_2D = plt.subplots(2, figsize=(25, 25))
    fig.suptitle(title, fontsize=64)
    ax_2D[0].set_title("x", fontdict={'fontsize': 36})
    ax_2D[1].set_title("y", fontdict={'fontsize': 36})

    for i in range(2):
        a_list, s_list, v_list, v_target, t_where_v_is_target, s_target, t_target = a_list_2D[i], s_list_2D[i], v_list_2D[i], v_target_2D[i], t_where_v_is_target_2D[i], s_target_2D[i], t_target_2D[i]
        # plot
        ax = ax_2D[i]
        #ax.plot(t, a_list, 'm', linewidth=2.0)
        ax.plot(t_list, v_list, 'g', linewidth=2.0)
        ax.plot(t_list, s_list, 'b', linewidth=2.0)
        ax.set(xlim=(0, 12), xticks=np.arange(0, 12),
               ylim=(-8, 9), yticks=np.arange(-8, 9))
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.axhline(y=s_target, color='b', linestyle=':')
        if t_target:
            ax.axvline(x=t_target, color='b', linestyle=':')
        ax.axhline(y=v_target, color='g', linestyle=':')
        if t_where_v_is_target:
            ax.axvline(x=t_where_v_is_target, color='g', linestyle=':')

    plt.show()