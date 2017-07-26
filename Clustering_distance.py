# !/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import nibabel as nib
from dipy.viz import actor, window, widget, fvtk
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
import nibabel.streamlines.tck as nibtck

# load brain_mask data
img = nib.load('/home/brain/workingdir/data/dwi/hcp/preprocessed/'
               'response_dhollander/100206/Structure/T1w_acpc_dc_restore_brain.nii.gz')
data = img.get_data()
shape = data.shape
affine = img.affine

# load fiber data
imgtck = nibtck.TckFile.load("/home/brain/workingdir/data/dwi/hcp/preprocessed/"
                             "response_dhollander/100206/result/CC_fib.tck")
streams = imgtck.streamlines
# print len(streams)
streamlines = [s for s in streams]

# print streamlines

world_coords = True
if not world_coords:
    from dipy.tracking.streamline import transform_streamlines
    streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

# clustering
qb = QuickBundles(threshold=20.)  # maybe 10.
clusters = qb.cluster(streamlines)

# extract > 100
# print len(clusters) # 294
for c in clusters:
    if len(c) < 100:
        clusters.remove_cluster(c)
# print len(clusters)  # 161

# print "Nb. clusters:", len(clusters)
# print "Cluster size:", map(len, clusters)
# print "Small clusters:", clusters < 10
# print "Streamlines indices of the first cluster:\n", clusters[0].indices
# print "Centroid of the last cluster:\n", clusters[-1].centroid

# show rhe initial dataset
ren = fvtk.ren()
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white))
fvtk.record(ren, n_frames=1, out_path='/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                                      'response_dhollander/100206/result/CC_fib1_20_100.png', size=(600, 600))

# show the centroids of the CC
colormap = fvtk.create_colormap(np.arange(len(clusters)))
fvtk.clear(ren)
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white, opacity=0.05))
fvtk.add(ren, fvtk.streamtube(clusters.centroids, colormap, linewidth=0.4))
fvtk.record(ren, n_frames=1, out_path='/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                                      'response_dhollander/100206/result/CC_fib2_20_100.png', size=(600, 600))

# show the label CC (colors form centroids)
colormap_full = np.ones((len(streamlines)), np.float64(3))
for clusters, color in zip(clusters, colormap):
    colormap_full[clusters.indices] = color
fvtk.clear(ren)
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, colormap_full))
fvtk.record(ren, n_frames=1, out_path='/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                                      'response_dhollander/100206/result/CC_fib3_20_100.png', size=(600, 600))
fvtk.show(ren)

# save the complete ClusterMap object with picking
save_pickle('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/100206/result/CC_fib_20_100.pk2', clusters)
