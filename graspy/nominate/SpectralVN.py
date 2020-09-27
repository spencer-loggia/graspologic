from abc import ABC

from .base import BaseVN
from ..embed import BaseEmbed
from ..embed import AdjacencySpectralEmbed as ase
import numpy as np
from scipy.spatial import distance
from scipy.stats import mode


class SpectralVN(BaseVN):

    def __init__(self, multigraph=False, embedding=None, embeder='ASE', mode='single_vertex'):
        super().__init__(multigraph=multigraph)
        if issubclass(type(embeder), BaseEmbed):
            self.embeder = embeder
        elif embeder == 'ASE':
            self.embeder = ase()
        else:
            raise TypeError
        self.embedding = embedding
        self.mode = mode
        self.distance_matrix = None
        self._attr_labels = None

    def _embed(self, X):
        X = np.array(X)

        # ensure X matches required dimensions for single and multigraph
        if self.multigraph and (len(X.shape) < 3 or X.shape[0] <= 1):
            raise IndexError("Argument must have dim 3")
        if not self.multigraph and len(X.shape) != 2:
            if len(X.shape) == 3 and X.shape[0] <= 1:
                X = X.reshape(X.shape[1], X.shape[2])
            else:
                raise IndexError("Argument must have dim 2")

        # Embed graph if embedding not provided
        if self.embedding is None:
            self.embedding = self.embeder.fit_transform(X)

    def _pairwise_dist(self, y:np.ndarray) -> np.ndarray:
        # y should give indexes
        y_vec = self.embedding[y[:, 0]]
        dist_mat = distance.cdist(self.embedding, y_vec)
        return dist_mat

    def fit(self, X, y):
        '''
        Constructs the embedding if needed.
        Parameters
        ----------
        X
        y: List of seed vertex indices, OR List of tuples of seed vertex indices and associated attributes.

        Returns
        -------

        '''
        if self.embedding is None:
            self._embed(X)

        # detect if y is attributed
        y = np.array(y)
        if np.ndim(y) < 2:
            y = y.reshape(1, 2)
        else:
            y = y.reshape(-1, 2)
        self.distance_matrix = self._pairwise_dist(y)
        self._attr_labels = y[:, 1]

    def predict(self):
        if self.mode == 'single_vertex':
            ordered = self.distance_matrix.argsort(axis=1)
            sorted_dists = self.distance_matrix[np.arange(ordered.shape[0], ordered.T)].T
            return ordered, sorted_dists
        elif self.mode == 'k-nearest':
            ordered = self.distance_matrix.argsort(axis=1)
            sorted_dists = self.distance_matrix[np.arange(ordered.shape[0]), ordered.T].T
            atts = self._attr_labels[ordered[:, :5]]
            att_preds, counts = mode(atts, axis=1)
            tile = np.tile(att_preds, (1, atts.shape[1]))
            inds = np.argwhere(atts == tile)
            place_hold = np.empty(atts.shape)
            place_hold[:] = np.NaN
            place_hold[inds[:, 0], inds[:, 1]] = sorted_dists[inds[:, 0], inds[:, 1]]
            pred_weights = np.nanmean(place_hold, axis=1)
            vert_order = np.argsort(pred_weights, axis=0)
            prediction = np.concatenate((vert_order.reshape(-1, 1), att_preds), axis=1)
            return prediction, pred_weights[vert_order]
        else:
            raise KeyError("no such mode " + str(self.mode))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.predict()







