# !/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np

def xmin_extract(streams):
    '''
    extract node according to x_min
    :param streams: streamlines img
    :return: extracted node
    '''
    Ls_temp = []
    for i in range(len(streams.streamlines)):
        l = []
        for j in range(len(streams.streamlines[i])):
            l.append(streams.streamlines[i][j][0])
        count = True
        for k in range(len(l) - 1):
            if l[k] * l[k + 1] < 0 and count:
                if np.abs(l[k]) < np.abs(l[k + 1]):
                    Ls_temp.append(streams.streamlines[i][k])
                else:
                    Ls_temp.append(streams.streamlines[i][k + 1])
                count = False
    return Ls_temp
