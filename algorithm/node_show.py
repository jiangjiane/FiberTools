# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def show_2d_node(Ls_temp):
    fig, ax = plt.subplots()
    ax.plot(np.array(Ls_temp)[:, 1], np.array(Ls_temp)[:, 2], 'o')
    ax.set_title('y_z distribution')
    plt.show()

def show_dist_matrix(sdist):
    name = range(sdist.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(sdist, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, sdist.shape[0], 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(name)
    ax.set_yticklabels(name)
    plt.show()
