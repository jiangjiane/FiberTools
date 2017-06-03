# !/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import nipype.interfaces.mrtrix as mrt

# tck2trk
tck2trk = mrt.MRTrix2TrackVis()
tck2trk.inputs.in_file = '/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/Diffusion/l250_hcp_FFA_projabs-0-abs-2-1.tck'
tck2trk.inputs.image_file = '/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/Diffusion/data.nii.gz'
tck2trk.inputs.out_filename = '/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/Diffusion/l250_hcp_FFA_projabs-0-abs-2-1.trk'
tck2trk.run()
