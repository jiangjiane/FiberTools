# !/usr/bin/python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np

label = nib.load('/home/brain/workingdir/HCP_label/brain_label/fsaverage.label.R.164k_fsavg_R.label.gii')
label_data = label.darrays
labels = label_data[0].data

# print labels

geo = nib.freesurfer.read_geometry('/home/brain/workingdir/HCP_label/brain_label/lh.white')
geo_coords, geo_faces = geo

# print geo_coords
# print geo_faces
FFA_label = []
FFA_coords = []
for i in range(len(labels)):
    if labels[i] == 22:
        FFA_label.append(i)
        FFA_coords.append(list(geo_coords[i]))

# print FFA_label
# print FFA_coords

for j in range(len(FFA_label)):
    FFA_coords[j].insert(0, FFA_label[j])
    FFA_coords[j].append(float(0))

FFA_list = FFA_coords
print FFA_list
FFA_array = np.asarray(FFA_list)
print FFA_array
print FFA_array.shape

np.savetxt('rh_OFA.label', FFA_list, fmt=['%d', '%f', '%f', '%f', '%f'], header=str(FFA_array.shape[0]), comments='#!ascii label  , from subject  vox2ras=TkReg\n')