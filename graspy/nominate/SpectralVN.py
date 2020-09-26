from .base import BaseVN
from ..embed import BaseEmbed
from ..embed import AdjacencySpectralEmbed as ase
import numpy as np
from scipy.spatial import distance

class SpectralVN(BaseVN):

    def __init__(self, multigraph=False, embedding=None, embeder='ASE', mode='euclidian'):
        super().__init__(multigraph=multigraph)
        if issubclass(type(embeder), BaseEmbed):
            self.embeder = embeder
        elif embeder == 'ASE':
            self.embeder = ase()
        else:
            raise TypeError
        self.embedding = embedding
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
        if self.embedding is not None:
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
            self.embedding = self._embed(X)

        # detect if y is attributed
        y = np.array(y)
        self.distance_matrix = self._pairwise_dist(y)
        self._attr_labels = y[:, 1]

    def predict(self):
        ordered = self.distance_matrix.T.argsort(axis=0)
        sorted_dists = self.distance_matrix[ordered, np.arange(ordered.shape[1])]
        return ordered, sorted_dists







