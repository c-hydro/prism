"""
Library Features:

Name:          libs_idw
Author(s):     Paul Brodersen (paulbrodersen+idw@gmail.com)
               Andrea Libertino (andrea.libertino@cimafoundation.org)
Date:          '20240520'
Version:       '1.0.0'
Reference:     Adapted from https://github.com/paulbrodersen/inverse_distance_weighting
"""

"""
Inverse distance weighting (IDW)
--------------------------------

Compute the score of query points based on the scores of their k-nearest neighbours,
weighted by the inverse of their distances.

"""

import numpy as np
from scipy.spatial import cKDTree


class tree(object):
    """
    Compute the score of query points based on the scores of their k-nearest neighbours,
    weighted by the inverse of their distances.

    @reference:
    https://en.wikipedia.org/wiki/Inverse_distance_weighting

    Arguments:
    ----------
        X: (N, d) ndarray
            Coordinates of N sample points in a d-dimensional space.
        z: (N,) ndarray
            Corresponding scores.
        leafsize: int (default 10)
            Leafsize of KD-tree data structure;
            should be less than 20.

    Returns:
    --------
        tree instance: object
    """
    def __init__(self, X=None, z=None, leafsize=10):
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize )
        if not z is None:
            self.z = np.array(z)

    def __call__(self, X, exp_idw=2, k=6, eps=1e-6, p=2, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.

            k: int (default 6)
                Number of nearest neighbours to use.

            exp_idw: int or inf
                Exponent of the inverse-distance interpolation.
                The higher the exponent, the more narrow is the interpolation

            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance

            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.

            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.

        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        """
        self.distances, self.idx = self.tree.query(X, k, eps=eps, p=p)
        self.distances += regularize_by
        weights = self.z[self.idx.ravel()].reshape(self.idx.shape)
        mw = np.sum(weights/(self.distances)**exp_idw, axis=1) / np.sum(1./(self.distances)**exp_idw, axis=1)
        return mw