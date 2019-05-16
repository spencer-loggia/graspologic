from .base import (
    BaseGraphEstimator,
    _calculate_p,
    _fit_weights,
    cartprod,
    _check_n_samples,
    _n_to_labels,
)
from ..utils import (
    import_graph,
    binarize,
    is_almost_symmetric,
    augment_diagonal,
    is_unweighted,
    symmetrize,
)
import numpy as np
from ..simulations import sbm, sample_edges
from ..cluster import GaussianCluster
from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted


def _check_common_inputs(n_components, min_comm, max_comm, cluster_kws, embed_kws):
    if not isinstance(n_components, int) and not n_components is None:
        raise TypeError("n_components must be an int or None")
    elif not n_components is None and n_components < 1:
        raise ValueError("n_components must be > 0")

    if not isinstance(min_comm, int):
        raise TypeError("min_comm must be an int")
    elif min_comm < 1:
        raise ValueError("min_comm must be > 0")

    if not isinstance(max_comm, int):
        raise TypeError("max_comm must be an int")
    elif max_comm < 1:
        raise ValueError("max_comm must be > 0")
    elif max_comm < min_comm:
        raise ValueError("max_comm must be >= min_comm")

    if not isinstance(cluster_kws, dict):
        raise TypeError("cluster_kws must be a dict")

    if not isinstance(embed_kws, dict):
        raise TypeError("embed_kws must be a dict")


class SBEstimator(BaseGraphEstimator):
    """
    Stochastic Block Model 

    The stochastic block model (SBM) represents each node as belonging to a block 
    (or community). For a given potential edge between node :math:`i` and :math:`j`, 
    the probability of an edge existing is specified by the block that nodes :math:`i`
    and :math:`j` belong to:

    ::math::`P_{ij} = B_\{tau_i}\{tau_j}`
    
    where :math:`B \in \mathbb{[0, 1]}^{K x K}` and :math:`\{tau}` is an :math:`n_nodes` 
    length vector specifying which block each node belongs to. 

    Parameters
    ----------
    directed : boolean, optional (default=True)
        Whether to treat the input graph as directed. Even if a directed graph is inupt, 
        this determines whether to force symmetry upon the block probability matrix fit
        for the SBM. It will also determine whether graphs sampled from the model are 
        directed. 
    loops : boolean, optional (default=False)
        Whether to allow entries on the diagonal of the adjacency matrix, i.e. loops in 
        the graph where a node connects to itself. 

    References
    ----------

    """

    def __init__(
        self,
        directed=True,
        loops=False,
        n_components=None,
        min_comm=1,
        max_comm=10,
        cluster_kws={},
        embed_kws={},
    ):
        super().__init__(directed=directed, loops=loops)

        _check_common_inputs(n_components, min_comm, max_comm, cluster_kws, embed_kws)

        self.cluster_kws = cluster_kws
        self.n_components = n_components
        self.min_comm = min_comm
        self.max_comm = max_comm
        self.embed_kws = embed_kws

    def _estimate_assignments(self, graph):
        """
        Do some kind of clustering algorithm to estimate communities

        There are many ways to do this, here is one
        """
        embed_graph = augment_diagonal(graph)
        latent = AdjacencySpectralEmbed(
            n_components=self.n_components, **self.embed_kws
        ).fit_transform(embed_graph)
        if isinstance(latent, tuple):
            latent = np.concatenate(latent, axis=1)
        gc = GaussianCluster(
            min_components=self.min_comm,
            max_components=self.max_comm,
            **self.cluster_kws
        )
        vertex_assignments = gc.fit_predict(latent)
        self.vertex_assignments_ = vertex_assignments

    def fit(self, graph, y=None):
        """
        Fit the SBM model to a graph, optionally with known block labels

        If y is `None`, the block assignments for each vertex will first be
        estimated.

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        y : array_like, length graph.shape[0], optional
            Categorical labels for the block assignments of the graph

        """
        graph = import_graph(graph)

        if not is_unweighted(graph):
            raise NotImplementedError(
                "Graph model is currently only implemented for unweighted graphs."
            )

        if y is None:
            self._estimate_assignments(graph)
            y = self.vertex_assignments_

            _, counts = np.unique(y, return_counts=True)
            self.block_weights_ = counts / graph.shape[0]
        else:
            check_X_y(graph, y)

        block_vert_inds, block_inds, block_inv = _get_block_indices(y)

        block_p = _calculate_block_p(graph, block_inds, block_vert_inds)

        if not self.directed:
            block_p = symmetrize(block_p)
        self.block_p_ = block_p

        p_mat = _block_to_full(block_p, block_inv, graph.shape)
        if not self.loops:
            p_mat -= np.diag(np.diag(p_mat))
        self.p_mat_ = p_mat

        return self

    def _n_parameters(self):
        n_blocks = self.block_p_.shape[0]
        n_parameters = 0
        if self.directed:
            n_parameters += n_blocks ** 2
        else:
            n_parameters += n_blocks * (n_blocks + 1) / 2
        if hasattr(self, "vertex_assignments_"):
            n_parameters += n_blocks - 1
        return n_parameters

    def sample(self, n_samples=1):
        """
        Sample graphs (realizations) from the fitted model

        Can only be called after the the model has been fit 

        Parameters
        ----------
        n_samples : int (default 1), optional
            The number of graphs to sample 
        
        Returns 
        -------
        graphs : np.array (n_samples, n_verts, n_verts)
            Array of sampled graphs, where the first dimension 
            indexes each sample, and the other dimensions represent
            (n_verts x n_verts) adjacency matrices for the sampled graphs. 

            Note that if only one sample is drawn, a (1, n_verts, n_verts) 
            array will still be returned. 
        """
        if hasattr(self, "vertex_assignments_"):
            check_is_fitted(self, "p_mat_")
            _check_n_samples(n_samples)
            n_verts = self.p_mat_.shape[0]

            graphs = np.zeros((n_samples, n_verts, n_verts))
            for i in range(n_samples):
                block_proportions = np.random.multinomial(n_verts, self.block_weights_)
                block_inv = _n_to_labels(block_proportions)
                p_mat = _block_to_full(self.block_p_, block_inv, self.p_mat_.shape)
                graphs[i, :, :] = sample_edges(
                    p_mat, directed=self.directed, loops=self.loops
                )
            return graphs
        else:
            return super().sample(n_samples=n_samples)


class DCSBEstimator(BaseGraphEstimator):
    """
    Degree-corrected Stochastic Block Model

    Parameters
    ----------
    directed : boolean, optional (default=True)
        Whether to treat the input graph as directed. Even if a directed graph is inupt, 
        this determines whether to force symmetry upon the block probability matrix fit
        for the SBM. It will also determine whether graphs sampled from the model are 
        directed. 
    degree_directed : boolean, optional (default=False)
        Whether to fit an "in" and "out" degree correction for each node. In the
        degree_directed case, the fit model can have a different expected in and out 
        degree for each node. 
    loops : boolean, optional (default=False)
        Whether to allow entries on the diagonal of the adjacency matrix, i.e. loops in 
        the graph where a node connects to itself. 

    """

    def __init__(
        self,
        degree_directed=False,
        directed=True,
        loops=False,
        n_components=None,
        min_comm=1,
        max_comm=10,
        cluster_kws={},
        embed_kws={},
    ):
        super().__init__(directed=directed, loops=loops)
        _check_common_inputs(n_components, min_comm, max_comm, cluster_kws, embed_kws)

        if not isinstance(degree_directed, bool):
            raise TypeError("`degree_directed` must be of type bool")

        self.degree_directed = degree_directed
        self.cluster_kws = cluster_kws
        self.n_components = n_components
        self.min_comm = min_comm
        self.max_comm = max_comm
        self.embed_kws = {}

    def _estimate_assignments(self, graph):
        graph = symmetrize(graph, method="avg")
        lse = LaplacianSpectralEmbed(
            form="R-DAD", n_components=self.n_components, **self.embed_kws
        )
        latent = lse.fit_transform(graph)
        gc = GaussianCluster(
            min_components=self.min_comm,
            max_components=self.max_comm,
            **self.cluster_kws
        )
        self.vertex_assignments_ = gc.fit_predict(latent)

    def fit(self, graph, y=None):
        """
        Fit the DCSBM model to a graph, optionally with known block labels

        If y is `None`, the block assignments for each vertex will first be
        estimated.

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to fit

        y : array_like, length graph.shape[0], optional
            Categorical labels for the block assignments of the graph

        Returns
        -------
        self : DCSBEstimator object 
            Fitted instance of self 
        """
        graph = import_graph(graph)
        if y is None:
            self._estimate_assignments(graph)
            y = self.vertex_assignments_
            _, counts = np.unique(y, return_counts=True)
            self.block_weights_ = counts / graph.shape[0]

        block_vert_inds, block_inds, block_inv = _get_block_indices(y)
        block_p = _calculate_block_p(graph, block_inds, block_vert_inds)

        out_degree = np.count_nonzero(graph, axis=1).astype(float)
        in_degree = np.count_nonzero(graph, axis=0).astype(float)
        if self.degree_directed:
            degree_corrections = np.stack((out_degree, in_degree), axis=1)
        else:
            degree_corrections = (out_degree + in_degree) / 2
            # new axis just so we can index later
            degree_corrections = degree_corrections[:, np.newaxis]
        for i in block_inds:
            block_degrees = degree_corrections[block_vert_inds[i]]
            degree_divisor = np.sum(block_degrees, axis=0)
            if not isinstance(degree_divisor, np.float64):
                degree_divisor[degree_divisor == 0] = 1
            degree_corrections[block_vert_inds[i]] = (
                degree_corrections[block_vert_inds[i]] / degree_divisor
            )
        self.degree_corrections_ = degree_corrections

        block_p = _calculate_block_p(
            graph, block_inds, block_vert_inds, return_counts=True
        )
        p_mat = _block_to_full(block_p, block_inv, graph.shape)
        p_mat = p_mat * np.outer(degree_corrections[:, 0], degree_corrections[:, -1])

        if not self.loops:
            p_mat -= np.diag(np.diag(p_mat))
        self.p_mat_ = p_mat
        self.block_p_ = block_p
        return self

    def _n_parameters(self):
        n_blocks = self.block_p_.shape[0]
        n_parameters = 0
        if self.directed:
            n_parameters += n_blocks ** 2  # B matrix
        else:
            n_parameters += n_blocks * (n_blocks + 1) / 2  # Undirected B matrix
        if hasattr(self, "vertex_assignments_"):
            # TODO other models where we sample a block comm and a dc
            n_parameters += self.vertex_assignments_
        n_parameters += self.degree_corrections_.size
        return n_parameters

    def sample(self, n_samples=1):
        """
        Sample graphs (realizations) from the fitted model

        Can only be called after the the model has been fit 

        Parameters
        ----------
        n_samples : int (default 1), optional
            The number of graphs to sample 
        
        Returns 
        -------
        graphs : np.array (n_samples, n_verts, n_verts)
            Array of sampled graphs, where the first dimension 
            indexes each sample, and the other dimensions represent
            (n_verts x n_verts) adjacency matrices for the sampled graphs. 

            Note that if only one sample is drawn, a (1, n_verts, n_verts) 
            array will still be returned. 
        """
        if hasattr(self, "vertex_assignments_"):
            check_is_fitted(self, "p_mat_")
            _check_n_samples(n_samples)
            n_verts = self.p_mat_.shape[0]

            graphs = np.zeros((n_samples, n_verts, n_verts))
            for i in range(n_samples):
                block_proportions = np.random.multinomial(n_verts, self.block_weights_)
                block_inv = _n_to_labels(block_proportions)
                p_mat = _block_to_full(self.block_p_, block_inv, self.p_mat_.shape)
                graphs[i, :, :] = sample_edges(
                    p_mat, directed=self.directed, loops=self.loops
                )
            return graphs
        else:
            return super().sample(n_samples=n_samples)


def _get_block_indices(y):
    """
    y is a length n_verts vector of labels

    returns a length n_verts vector in the same order as the input
    indicates which block each node is
    """
    block_labels, block_inv, block_sizes = np.unique(
        y, return_inverse=True, return_counts=True
    )

    n_blocks = len(block_labels)
    block_inds = range(n_blocks)

    block_vert_inds = []
    for i in block_inds:
        # get the inds from the original graph
        inds = np.where(block_inv == i)[0]
        block_vert_inds.append(inds)
    return block_vert_inds, block_inds, block_inv


def _calculate_block_p(graph, block_inds, block_vert_inds, return_counts=False):
    """
    graph : input n x n graph 
    block_inds : list of length n_communities
    block_vert_inds : list of list, for each block index, gives every node in that block
    return_counts : whether to calculate counts rather than proportions
    """

    n_blocks = len(block_inds)
    block_pairs = cartprod(block_inds, block_inds)
    block_p = np.zeros((n_blocks, n_blocks))

    for p in block_pairs:
        from_block = p[0]
        to_block = p[1]
        from_inds = block_vert_inds[from_block]
        to_inds = block_vert_inds[to_block]
        block = graph[from_inds, :][:, to_inds]
        if return_counts:
            p = np.count_nonzero(block)
        else:
            p = _calculate_p(block)
        block_p[from_block, to_block] = p
    return block_p


def _block_to_full(block_mat, inverse, shape):
    """
    "blows up" a k x k matrix, where k is the number of communities, 
    into a full n x n probability matrix

    block mat : k x k 
    inverse : array like length n, 
    """
    block_map = cartprod(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat
