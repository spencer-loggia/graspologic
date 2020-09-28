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

    def predict(self, out="best_preds"):
        if self.mode == 'single_vertex':
            ordered = self.distance_matrix.argsort(axis=1)
            sorted_dists = self.distance_matrix[np.arange(ordered.shape[0], ordered.T)].T
            return ordered, sorted_dists

        elif self.mode == 'knn-weighted':
            ordered = self.distance_matrix.argsort(axis=1)
            sorted_dists = self.distance_matrix[np.arange(ordered.shape[0]), ordered.T].T
            atts = self._attr_labels[ordered[:, :5]]
            unique_att = np.unique(atts)
            pred_weights = np.empty((atts.shape[0], unique_att.shape[0])) # use this array for bin counts as well to save space
            for i in range(unique_att.shape[0]):
                pred_weights[:, i] = np.count_nonzero(atts == unique_att[i], axis=1)
                inds = np.argwhere(atts == unique_att[i])
                place_hold = np.empty(atts.shape)
                place_hold[:] = np.NaN
                place_hold[inds[:, 0], inds[:, 1]] = sorted_dists[inds[:, 0], inds[:, 1]]
                pred_weights[:, i] = np.nansum(place_hold, axis=1) / np.power(pred_weights[:, i], 2)

            if out == 'best_preds':
                best_pred_inds = np.nanargmin(pred_weights, axis=1)
                best_pred_weights = pred_weights[np.arange(pred_weights.shape[0]), best_pred_inds]
                vert_order = np.argsort(best_pred_weights, axis=0)
                att_preds = unique_att[best_pred_inds[vert_order]]
                prediction = np.concatenate((vert_order.reshape(-1, 1), att_preds.reshape(-1, 1)), axis=1)
                return prediction, pred_weights[vert_order]
            elif out == 'per_attribute':
                pred_weights[np.argwhere(np.isnan(pred_weights))] = np.nanmax(pred_weights)
                vert_orders = np.argsort(pred_weights, axis=0)
                return vert_orders, pred_weights[vert_orders]
        else:
            raise KeyError("no such mode " + str(self.mode))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.predict()







