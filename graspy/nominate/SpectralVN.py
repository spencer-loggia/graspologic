from typing import Union
from abc import abstractmethod
from .base import BaseVN
from ..embed import BaseEmbed
from ..embed import AdjacencySpectralEmbed as ase
import numpy as np
from scipy.spatial import distance
from scipy.stats import mode

class BaseSpectralVN(BaseVN):
    def __init__(self, multigraph: bool, embedding: np.ndarray, embeder: Union[str, BaseEmbed], mode: str):
        super().__init__(multigraph=multigraph)
        self.embedding = embedding
        if self.embedding is None:
            if issubclass(type(embeder), BaseEmbed):
                self.embeder = embeder
            elif embeder == 'ASE':
                self.embeder = ase()
            else:
                raise TypeError
        self._attr_labels = None
        self.mode = mode

    def _embed(self, X: np.ndarray):
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

    def fit(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike):
        """
        Constructs the embedding if needed.
        Parameters
        ----------
        X
        y: List of seed vertex indices, OR List of tuples of seed vertex indices and associated attributes.

        Returns
        -------

        """
        X = np.array(X)
        if self.embedding is None:
            self._embed(X)

        # detect if y is attributed
        y = np.array(y)
        if np.ndim(y) < 2:
            y = y.reshape(1, 2)
        else:
            y = y.reshape(-1, 2)
        self._attr_labels = y[:, 1]

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def fit_transform(self, X, y=None):
        pass


class SpectralVertexNominator(BaseSpectralVN):

    def __init__(self, multigraph: bool = False,
                 embedding: np.ndarray = None,
                 embeder: Union[str, BaseEmbed] = 'ASE',
                 mode: str = 'single_vertex'):
        super(SpectralVertexNominator,
              self).__init__(multigraph=multigraph,
                             embedding=embedding,
                             embeder=embeder,
                             mode=mode)
        self.distance_matrix = None

    def _pairwise_dist(self, y: np.ndarray, metric='euclidian') -> np.ndarray:
        # wrapper for scipy's cdist function
        # y should give indexes
        y_vec = self.embedding[y[:, 0]]
        dist_mat = distance.cdist(self.embedding, y_vec, metric=metric)
        return dist_mat

    def fit(self, X, y):
        super().fit(X, y)
        self.distance_matrix = self._pairwise_dist(y)

    def _knn_simple_predict(self):
        '''
        Simplest possible methdod, doesn't consider attributes.
        If multiple seed vertices are provied, provides the top
        nominations for each individual seed.
        Returns
        -------

        '''
        ordered = self.distance_matrix.argsort(axis=1)
        sorted_dists = self.distance_matrix[np.arange(ordered.shape[0], ordered.T)].T
        return ordered, sorted_dists

    def _knn_weighted_predict(self, out, k=5):
        '''
        Nominate vertex based on distance from the k nearest neighbors of each class.
        The default decision function is sum(dist to each neighbor of class c) / (number_such_neighbors)^2.
        This is a good metric becuase it accounts for both number of neighbors from a class and their respective
        distances. However, assumes that all possible attributes are represented in the seed population.

        Parameters
        ----------
        out

        Returns
        -------

        '''
        ordered = self.distance_matrix.argsort(axis=1)
        sorted_dists = self.distance_matrix[np.arange(ordered.shape[0]), ordered.T].T
        atts = self._attr_labels[ordered[:, :k]]
        unique_att = np.unique(atts)
        pred_weights = np.empty(
            (atts.shape[0], unique_att.shape[0]))  # use this array for bin counts as well to save space
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

    def _kmeans_unsupervised_predict(self):
        '''
        Unsupervised kmeans spectral nomination, as described
        by Fishkind et. al. in Vertex Nomination for Membership Prediction.
        Has the advantage of being able to identify attributes not represented
        in seed population, however by default will select number of clusters
        based on number of unique attributes in given seed.
        Returns
        -------

        '''
        from sklearn.cluster import KMeans
        unique_att = np.unique(self._attr_labels)
        clf = KMeans(n_clusters=unique_att.shape[0])
        y_hat = clf.fit_transform(self.embedding)
        # now order for each cluster based of distance from centroid
        centroids = clf.cluster_centers_

    def predict(self, out="best_preds"):
        if self.mode == 'single_vertex':
            self._knn_simple_predict()
        elif self.mode == 'knn-weighted':
            self._knn_weighted_predict(out)
        else:
            raise KeyError("no such mode " + str(self.mode))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.predict()


class SpectralClusteringVN(BaseVN):
    def __init__(self):
        super().__init__(multigraph=True)
        pass
