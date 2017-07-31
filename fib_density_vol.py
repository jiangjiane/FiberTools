# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import dipy.tracking.utils as ditu
import nibabel.streamlines.tck as nibtck

img = nib.load("/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/Structure/T1w_short.nii.gz")
data = img.get_data()
shape = img.shape
# print img.affine
# print shape

img_cc = nibtck.TckFile.load('/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                             'response_dhollander/100206/result/CC_fib.tck')
streamstck = img_cc.streamlines
# print streamstck

image_volume = ditu.density_map(streamstck, vol_dims=shape, voxel_size=0.625, affine=img.affine)

dm_img = nib.Nifti1Image(image_volume.astype("int16"), img.affine)
dm_img.to_filename('/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                             'response_dhollander/100206/result/CC_fib_density_map.nii.gz')

