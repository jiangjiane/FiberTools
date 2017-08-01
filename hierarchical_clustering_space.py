# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from nibabel import streamlines
import nibabel.streamlines.tck as nibtck
import nibabel.streamlines.array_sequence as nibAS
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering


img_cc = nibtck.TckFile.load('/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                             'response_dhollander/100206/result/CC_fib.tck')
streamstck = img_cc.streamlines
# print streamstck[0]

L_temp_0 = nibAS.ArraySequence()
L_temp_1 = nibAS.ArraySequence()
L_temp_2 = nibAS.ArraySequence()
L_temp_3 = nibAS.ArraySequence()
Ls_temp = []

# remove non_CC
for i in range(len(img_cc.streamlines)):
    l_x = []
    for j in range(len(img_cc.streamlines[i])):
        l_x.append(np.abs(img_cc.streamlines[i][j][0]))
    x_min_index = np.argmin(l_x)
    Ls_temp.append(img_cc.streamlines[i][x_min_index])
        # pass
print Ls_temp[0]

connectivity = kneighbors_graph(Ls_temp, n_neighbors=10, include_self=False)
clusters = AgglomerativeClustering(n_clusters=4, connectivity=connectivity, linkage='average')
labels = clusters.fit_predict(Ls_temp)
print len(labels)
d = zip(labels, Ls_temp)
# print d[0]
# print d[1]
# print d[0][0]
# print d[0][1]

for k in d:
    if k[0] == 0:
        for m in img_cc.streamlines:
            if k[1] in m:
                L_temp_0.append(m)
    if k[0] == 1:
        for m in img_cc.streamlines:
            if k[1] in m:
                L_temp_1.append(m)
    if k[0] == 2:
        for m in img_cc.streamlines:
            if k[1] in m:
                L_temp_2.append(m)
    else:
        for m in img_cc.streamlines:
            if k[1] in m:
                L_temp_3.append(m)

tractogram = streamlines.tractogram.Tractogram(streamlines=L_temp_0, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat = nibtck.TckFile(tractogram=tractogram, header=img_cc.header)
datdat.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib0.tck')

tractogram1 = streamlines.tractogram.Tractogram(streamlines=L_temp_1, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat1 = nibtck.TckFile(tractogram=tractogram1, header=img_cc.header)
datdat1.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib1.tck')

tractogram2 = streamlines.tractogram.Tractogram(streamlines=L_temp_2, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat2 = nibtck.TckFile(tractogram=tractogram2, header=img_cc.header)
datdat2.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib2.tck')

tractogram3 = streamlines.tractogram.Tractogram(streamlines=L_temp_3, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat3 = nibtck.TckFile(tractogram=tractogram3, header=img_cc.header)
datdat3.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib3.tck')
