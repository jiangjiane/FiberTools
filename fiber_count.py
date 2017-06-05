# !/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from nibabel import trackvis
from dipy.tracking.utils import length
import matplotlib.pyplot as plt


fname = '/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/lr250_sift12_hcp_FFA_projabs-2.trk'
streams, hdr = trackvis.read(fname, points_space="rasmm")
streamlines = [s[0] for s in streams]
# print len(list(length(streamlines)))

# length
lengths = list(length(streamlines))

# show
plt.figure('Fiber statistics')
plt.subplot(111)
plt.title('Length histogram')
plt.hist(lengths, color='burlywood')
plt.xlabel('Length')
plt.ylabel('Count')

# save length histogram
plt.legend()
plt.savefig('lr250_sift12_hcp_FFA_projabs-2_length_histogram.png')

plt.show()
