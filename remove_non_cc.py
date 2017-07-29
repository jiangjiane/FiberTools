# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from nibabel import streamlines
import nibabel.streamlines.tck as nibtck
import nibabel.streamlines.array_sequence as nibAS


# load tck data
img_cc = nibtck.TckFile.load('/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                             'response_dhollander/100206/result/CC_fib.tck')
streamstck = img_cc.streamlines
# print streamstck[0][0][0]
# print type(streamstck)

L_temp = nibAS.ArraySequence()

# remove non_CC
for i in range(len(img_cc.streamlines)):
    l_x = []
    for j in range(len(img_cc.streamlines[i])):
        l_x.append(np.abs(img_cc.streamlines[i][j][0]))
    x_min_index = np.argmin(l_x)
    if img_cc.streamlines[i][x_min_index][2] > -10:  # -2<x<2 & z>-10
        L_temp.append(img_cc.streamlines[i])
        # pass
print L_temp

tractogram = streamlines.tractogram.Tractogram(streamlines=L_temp, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat = nibtck.TckFile(tractogram=tractogram, header=img_cc.header)
datdat.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/'
            'response_dhollander/100206/result/CC_fib_remove_non_cc_z-10_x-min.tck')
