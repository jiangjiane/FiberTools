# !/usr/bin/python
# -*- coding: utf-8 -*-


from nibabel import streamlines
import nibabel.streamlines.tck as nibtck
import nibabel.streamlines.trk as nibtrk
import nibabel.streamlines.array_sequence as nibAS


imgtck = nibtck.TckFile.load("/home/brain/workingdir/data/dwi/hcp/preprocessed/"
                             "response_dhollander/100206/Diffusion/100k_sift_1M45006_dynamic250.tck")
# imgtck = streamlines.load("/home/brain/workingdir/data/dwi/hcp/preprocessed/
# response_dhollander/100206/Diffusion/100k_sift_1M45006_dynamic250.tck")
# streamstck = imgtck.streamlines
# print streamstck
# print type(streamstck)

# imgtrk =nibtrk.TrkFile.load("/home/brain/workingdir/data/dwi/hcp/preprocessed
# /response_dhollander/100206/Diffusion/100k_sift_1M45006_dynamic250.trk")
# streamstrk = imgtrk.streamlines
# print streamstrk[0]

L_temp = nibAS.ArraySequence()

# extract CC
for i in range(len(imgtck.streamlines)):
    if imgtck.streamlines[i][0][0]*imgtck.streamlines[i][-1][0] < 0:
        L_temp.append(imgtck.streamlines[i])
        # pass
# print L_temp
tractogram = streamlines.tractogram.Tractogram(streamlines=L_temp, data_per_streamline=imgtck.tractogram.data_per_streamline,
                                               data_per_point=imgtck.tractogram.data_per_point, affine_to_rasmm=imgtck.tractogram.affine_to_rasmm)
datdat = nibtck.TckFile(tractogram=tractogram, header=imgtck.header)
datdat.save("/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib.tck")
