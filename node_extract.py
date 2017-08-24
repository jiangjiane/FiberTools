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
# print streamstck[0][0]
# print streamstck[0][0][0]
print len(streamstck)

L_temp_0 = nibAS.ArraySequence()
L_temp_1 = nibAS.ArraySequence()
L_temp_2 = nibAS.ArraySequence()
L_temp_3 = nibAS.ArraySequence()
Ls_temp = []

# remove non_CC
for i in range(len(img_cc.streamlines)):
    l = []
    for j in range(len(img_cc.streamlines[i])):
        l.append(img_cc.streamlines[i][j][0])
    count = True
    for k in range(len(l)-1):
        if l[k] * l[k+1] < 0 and count:
            if np.abs(l[k]) < np.abs(l[k+1]):
                Ls_temp.append(img_cc.streamlines[i][k])
            else:
                Ls_temp.append(img_cc.streamlines[i][k+1])
            count = False

print Ls_temp[0]
print len(Ls_temp)

connectivity = kneighbors_graph(Ls_temp, n_neighbors=30, mode='connectivity', include_self=True)
clusters = AgglomerativeClustering(n_clusters=4, connectivity=connectivity, linkage='average')
labels = clusters.fit_predict(Ls_temp)
print len(labels)
d = zip(labels, Ls_temp)

# print d[0]
# print d[1]
# print d[0][0]
# print d[0][1]
print len(d)
# print type(d)

for k in range(len(d)):
    if d[k][0] == 0:
        L_temp_0.append(img_cc.streamlines[k])
    if d[k][0] == 1:
        L_temp_1.append(img_cc.streamlines[k])
    if d[k][0] == 2:
        L_temp_2.append(img_cc.streamlines[k])
    else:
        L_temp_3.append(img_cc.streamlines[k])

tractogram = streamlines.tractogram.Tractogram(streamlines=L_temp_0, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat = nibtck.TckFile(tractogram=tractogram, header=img_cc.header)
datdat.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib_only2_0.tck')

tractogram1 = streamlines.tractogram.Tractogram(streamlines=L_temp_1, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat1 = nibtck.TckFile(tractogram=tractogram1, header=img_cc.header)
datdat1.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib_only2_1.tck')

tractogram2 = streamlines.tractogram.Tractogram(streamlines=L_temp_2, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat2 = nibtck.TckFile(tractogram=tractogram2, header=img_cc.header)
datdat2.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib_only2_2.tck')

tractogram3 = streamlines.tractogram.Tractogram(streamlines=L_temp_3, data_per_streamline=img_cc.tractogram.data_per_streamline,
                                               data_per_point=img_cc.tractogram.data_per_point, affine_to_rasmm=img_cc.tractogram.affine_to_rasmm)
datdat3 = nibtck.TckFile(tractogram=tractogram3, header=img_cc.header)
datdat3.save('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib_only2_3.tck')