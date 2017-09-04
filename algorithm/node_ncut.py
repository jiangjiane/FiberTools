# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from rw.load import load_tck
from rw.save import save_tck
from node_extract import xmin_extract
from ncut import ncut, discretisation, get_labels
from node_show import show_2d_node, show_dist_matrix
from metric import coordinate_dist

import nibabel.streamlines.array_sequence as nibAS

import matplotlib.pyplot as plt

# load data
tck_path = '/home/brain/workingdir/data/dwi/hcp/' \
           'preprocessed/response_dhollander/100206/result/CC_fib.tck'
imgtck = load_tck(tck_path)
# streamstck = img_cc.streamlines

# extract node according to x-value
Ls_temp = xmin_extract(imgtck)
show_2d_node(Ls_temp)

# calculate similarity matrix
sdist = coordinate_dist(Ls_temp)
print sdist

# set the correlation matrix
thre0 = sdist > 5.6
sdist[thre0] = 0
thre1 = sdist > 0
# sdist[thre1] = sdist[thre1] / sdist[thre1].max()
# sdist[thre1] = 1 - sdist[thre1]
sdist[thre1] = 1
print sdist

# show the sdist matrix
# show_dist_matrix(sdist)

# ncut according to coordinate
eigen_val, eigen_vec = ncut(sdist, 4)
eigenvec_discrete = discretisation(eigen_vec)
print eigenvec_discrete

# get labels
label_img = get_labels(eigenvec_discrete)

print label_img

# choose fiber according to node clusters
d = zip(label_img, Ls_temp)
L_temp_0 = nibAS.ArraySequence()
L_temp_1 = nibAS.ArraySequence()
L_temp_2 = nibAS.ArraySequence()
L_temp_3 = nibAS.ArraySequence()
L_temp0 = []
L_temp1 = []
L_temp2 = []
L_temp3 = []

for k in range(len(d)):
    if d[k][0] == 0:
        L_temp_0.append(imgtck.streamlines[k])
        L_temp0.append(d[k][1])
    if d[k][0] == 1:
        L_temp_1.append(imgtck.streamlines[k])
        L_temp1.append(d[k][1])
    if d[k][0] == 2:
        L_temp_2.append(imgtck.streamlines[k])
        L_temp2.append(d[k][1])
    if d[k][0] == 3:
        L_temp_3.append(imgtck.streamlines[k])
        L_temp3.append(d[k][1])

# show node clusters
fig, ax = plt.subplots()
ax.plot(np.array(L_temp0)[:, 1], np.array(L_temp0)[:, 2], 'o', color='r')
ax.plot(np.array(L_temp1)[:, 1], np.array(L_temp1)[:, 2], 'o', color='b')
ax.plot(np.array(L_temp2)[:, 1], np.array(L_temp2)[:, 2], 'o', color='g')
ax.plot(np.array(L_temp3)[:, 1], np.array(L_temp3)[:, 2], 'o', color='c')
plt.show()

# save the fiber cluster
out_path = '/home/brain/workingdir/data/dwi/hcp/' \
           'preprocessed/response_dhollander/100206/result/CC_fib_ncut5_set0-1_0.tck'
out_path1 = '/home/brain/workingdir/data/dwi/hcp/' \
            'preprocessed/response_dhollander/100206/result/CC_fib_ncut5_set0-1_1.tck'
out_path2 = '/home/brain/workingdir/data/dwi/hcp/' \
            'preprocessed/response_dhollander/100206/result/CC_fib_ncut5_set0-1_2.tck'
out_path3 = '/home/brain/workingdir/data/dwi/hcp/' \
            'preprocessed/response_dhollander/100206/result/CC_fib_ncut5_set0-1_3.tck'

save_tck(L_temp_0, imgtck.header, imgtck.tractogram.data_per_streamline,
         imgtck.tractogram.data_per_point, imgtck.tractogram.affine_to_rasmm, out_path)
save_tck(L_temp_1, imgtck.header, imgtck.tractogram.data_per_streamline,
         imgtck.tractogram.data_per_point, imgtck.tractogram.affine_to_rasmm, out_path1)
save_tck(L_temp_2, imgtck.header, imgtck.tractogram.data_per_streamline,
         imgtck.tractogram.data_per_point, imgtck.tractogram.affine_to_rasmm, out_path2)
save_tck(L_temp_3, imgtck.header, imgtck.tractogram.data_per_streamline,
         imgtck.tractogram.data_per_point, imgtck.tractogram.affine_to_rasmm, out_path3)
