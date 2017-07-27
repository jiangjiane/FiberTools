# !/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from dipy.viz import fvtk
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import SumPointwiseEuclideanMetric
from dipy.io.pickles import save_pickle
import nibabel.streamlines.tck as nibtck

from dipy.segment.metric import Feature, Metric, VectorOfEndpointsFeature
from dipy.tracking.streamline import length


# load streamline data
def load(filename):
    """Load streamlines data"""
    imgtck = nibtck.TckFile.load(filename)
    return imgtck

class ArcLengthFeature(Feature):
    """ Computes the arc length of a streamline. """
    def __init__(self):
        # The arc length stays the same even if the streamline is reversed.
        super(ArcLengthFeature, self).__init__(is_order_invariant=True)

    def infer_shape(self, streamline):
        """ Infers the shape of features extracted from `streamline`. """
        # Arc length is a scalar
        return 1

    def extract(self, streamline):
        """ Extracts features from `streamline`. """
        # return np.sum(np.sqrt(np.sum((streamline[1:] - streamline[:-1]) ** 2)))
        # or use a Dipy's function that computes the arc length of a streamline.
        return length(streamline)

# load fiber data
imgtck = load('/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                             'response_dhollander/101006/result/CC_fib.tck')
streams = imgtck.streamlines
# print len(streams)
streamlines = [s for s in streams]

# print streamlines

world_coords = True
if not world_coords:
    from dipy.tracking.streamline import transform_streamlines
    streamlines = transform_streamlines(streamlines, np.linalg.inv(imgtck.affine))

metric = SumPointwiseEuclideanMetric(feature=ArcLengthFeature())
qb = QuickBundles(threshold=2., metric=metric)
clusters = qb.cluster(streamlines)

# extract > 100
# print len(clusters) # 89
for c in clusters:
    if len(c) < 100:
        clusters.remove_cluster(c)

# print len(clusters)  # 70

# print "Nb. clusters:", len(clusters)
# print "Cluster size:", map(len, clusters)
# print "Small clusters:", clusters < 10
# print "Streamlines indices of the first cluster:\n", clusters[0].indices
# print "Centroid of the last cluster:\n", clusters[-1].centroid

# visualize the clustering result
# Color each streamline according to the cluster they belong to.
colormap = fvtk.create_colormap(np.ravel(clusters.centroids), name='jet')
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

ren = fvtk.ren()
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, colormap_full))
fvtk.record(ren, n_frames=1, out_path='/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                                      'response_dhollander/101006/result/CC_fib_length1_2.png', size=(600, 600))
# fvtk.show(ren)

class CosineMetric(Metric):
    """ Computes the cosine distance between two streamlines. """
    def __init__(self):
        # For simplicity, features will be the vector between endpoints of a streamline.
        super(CosineMetric, self).__init__(feature=VectorOfEndpointsFeature())

    def are_compatible(self, shape1, shape2):
        """ Checks if two features are vectors of same dimension.

        Basically this method exists so we don't have to do this check
        inside the `dist` method (speedup).
        """
        return shape1 == shape2 and shape1[0] == 1

    def dist(self, v1, v2):
        """ Computes a the cosine distance between two vectors. """
        norm = lambda x: np.sqrt(np.sum(x**2))
        cos_theta = np.dot(v1, v2.T) / (norm(v1)*norm(v2))

        # Make sure it's in [-1, 1], i.e. within domain of arccosine
        cos_theta = np.minimum(cos_theta, 1.)
        cos_theta = np.maximum(cos_theta, -1.)
        return np.arccos(cos_theta) / np.pi  # Normalized cosine distance

metric = CosineMetric()
qb = QuickBundles(threshold=0.1, metric=metric)
clusters = qb.cluster(streamlines)

# extract > 100
# print len(clusters)  # 41
for c in clusters:
    if len(c) < 100:
        clusters.remove_cluster(c)

# print len(clusters)  # 39

# visualize the clustering result.
# Color each streamline according to the cluster they belong to.
colormap = fvtk.create_colormap(np.arange(len(clusters)), name='jet')
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

ren = fvtk.ren()
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, colormap_full))
fvtk.record(ren, n_frames=1, out_path='/home/brain/workingdir/data/dwi/hcp/preprocessed/'
                                      'response_dhollander/101006/result/CC_fib_length2_2.png', size=(600, 600))
# fvtk.show(ren)

# save the complete ClusterMap object with picking
save_pickle('/home/brain/workingdir/data/dwi/hcp/preprocessed/response_dhollander/101006/result/CC_fib_length_2.pk2', clusters)