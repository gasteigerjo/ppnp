import warnings
from typing import Dict, Union, Tuple, Any
import numpy as np
import scipy.sparse as sp

__all__ = ['SparseGraph']

sparse_graph_properties = [
        'adj_matrix', 'attr_matrix', 'edge_attr_matrix', 'labels',
        'node_names', 'attr_names', 'edge_attr_names', 'class_names',
        'metadata']


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    All properties are immutable so users don't mess up the
    data format's assumptions (e.g. of edge_attr_matrix).
    Be careful when circumventing this and changing the internal matrices
    regardless (e.g. by exchanging the data array of a sparse matrix).

    Parameters
    ----------
    adj_matrix
        Adjacency matrix in CSR format. Shape [num_nodes, num_nodes]
    attr_matrix
        Attribute matrix in CSR or numpy format. Shape [num_nodes, num_attr]
    edge_attr_matrix
        Edge attribute matrix in CSR or numpy format. Shape [num_edges, num_edge_attr]
    labels
        Array, where each entry represents respective node's label(s). Shape [num_nodes]
        Alternatively, CSR matrix with labels in one-hot format. Shape [num_nodes, num_classes]
    node_names
        Names of nodes (as strings). Shape [num_nodes]
    attr_names
        Names of the attributes (as strings). Shape [num_attr]
    edge_attr_names
        Names of the edge attributes (as strings). Shape [num_edge_attr]
    class_names
        Names of the class labels (as strings). Shape [num_classes]
    metadata
        Additional metadata such as text.

    """
    def __init__(
            self, adj_matrix: sp.spmatrix,
            attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
            edge_attr_matrix: Union[np.ndarray, sp.spmatrix] = None,
            labels: Union[np.ndarray, sp.spmatrix] = None,
            node_names: np.ndarray = None,
            attr_names: np.ndarray = None,
            edge_attr_names: np.ndarray = None,
            class_names: np.ndarray = None,
            metadata: Any = None):
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)."
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree.")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)."
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree.")

        if edge_attr_matrix is not None:
            if sp.isspmatrix(edge_attr_matrix):
                edge_attr_matrix = edge_attr_matrix.tocsr().astype(np.float32)
            elif isinstance(edge_attr_matrix, np.ndarray):
                edge_attr_matrix = edge_attr_matrix.astype(np.float32)
            else:
                raise ValueError("Edge attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)."
                                 .format(type(edge_attr_matrix)))

            if edge_attr_matrix.shape[0] != adj_matrix.count_nonzero():
                raise ValueError("Number of edges and dimension of the edge attribute matrix don't agree.")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree.")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree.")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree.")

        if edge_attr_names is not None:
            if len(edge_attr_names) != edge_attr_matrix.shape[1]:
                raise ValueError("Dimensions of the edge attribute matrix and the edge attribute names don't agree.")

        self._adj_matrix = adj_matrix
        self._attr_matrix = attr_matrix
        self._edge_attr_matrix = edge_attr_matrix
        self._labels = labels
        self._node_names = node_names
        self._attr_names = attr_names
        self._edge_attr_names = edge_attr_names
        self._class_names = class_names
        self._metadata = metadata

        self._flag_writeable(False)

    def num_nodes(self) -> int:
        """Get the number of nodes in the graph.
        """
        return self.adj_matrix.shape[0]

    def num_edges(self, warn: bool = True) -> int:
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as _two_ edges.

        """
        if warn and not self.is_directed():
            warnings.warn("num_edges always returns the number of directed edges now.", FutureWarning)
        return self.adj_matrix.nnz

    def get_neighbors(self, idx: int) -> np.ndarray:
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def get_edgeid_to_idx_array(self) -> np.ndarray:
        """Return a Numpy Array that maps edgeids to the indices in the adjacency matrix.

        Returns
        -------
        np.ndarray
            The i'th entry contains the x- and y-coordinates of edge i in the adjacency matrix.
            Shape [num_edges, 2]

        """
        return np.transpose(self.adj_matrix.nonzero())

    def get_idx_to_edgeid_matrix(self) -> sp.csr_matrix:
        """Return a sparse matrix that maps indices in the adjacency matrix to edgeids.

        Caution: This contains one explicit 0 (zero stored as a nonzero),
        which is the index of the first edge.

        Returns
        -------
        sp.csr_matrix
            The entry [x, y] contains the edgeid of the corresponding edge (or 0 for non-edges).
            Shape [num_nodes, num_nodes]

        """
        return sp.csr_matrix(
                (np.arange(self.adj_matrix.nnz), self.adj_matrix.indices, self.adj_matrix.indptr),
                shape=self.adj_matrix.shape)

    def is_directed(self) -> bool:
        """Check if the graph is directed (adjacency matrix is not symmetric).
        """
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self) -> 'SparseGraph':
        """Convert to an undirected graph (make adjacency matrix symmetric).
        """
        idx = self.get_edgeid_to_idx_array().T
        ridx = np.ravel_multi_index(idx, self.adj_matrix.shape)
        ridx_rev = np.ravel_multi_index(idx[::-1], self.adj_matrix.shape)

        # Get duplicate edges (self-loops and opposing edges)
        dup_ridx = ridx[np.isin(ridx, ridx_rev)]
        dup_idx = np.unravel_index(dup_ridx, self.adj_matrix.shape)

        # Check if the adjacency matrix weights are symmetric (if nonzero)
        if not np.allclose(self.adj_matrix[dup_idx], self.adj_matrix[dup_idx[::-1]]):
            raise ValueError("Adjacency matrix weights of opposing edges differ.")

        # Create symmetric matrix
        new_adj_matrix = self.adj_matrix + self.adj_matrix.T
        new_adj_matrix[dup_idx] -= self.adj_matrix[dup_idx]
        flag_writeable(new_adj_matrix, False)

        if self.edge_attr_matrix is not None:

            # Check if edge attributes are symmetric
            edgeid_mat = self.get_idx_to_edgeid_matrix()
            dup_edgeids = edgeid_mat[dup_idx].A1
            dup_rev_edgeids = edgeid_mat[dup_idx[::-1]].A1
            if not np.allclose(self.edge_attr_matrix[dup_edgeids], self.edge_attr_matrix[dup_rev_edgeids]):
                raise ValueError("Edge attributes of opposing edges differ.")

            # Adjust edge attributes to new adjacency matrix
            edgeid_mat.data += 1  # Add 1 so we don't lose the explicit 0 and change the sparsity structure
            new_edgeid_mat = edgeid_mat + edgeid_mat.T
            new_edgeid_mat[dup_idx] -= edgeid_mat[dup_idx]
            new_idx = new_adj_matrix.nonzero()
            edgeids_perm = new_edgeid_mat[new_idx].A1 - 1
            self._edge_attr_matrix = self.edge_attr_matrix[edgeids_perm]
            flag_writeable(self._edge_attr_matrix, False)

        self._adj_matrix = new_adj_matrix
        return self

    def is_weighted(self) -> bool:
        """Check if the graph is weighted (edge weights other than 1).
        """
        return np.any(np.unique(self.adj_matrix[self.adj_matrix.nonzero()].A1) != 1)

    def to_unweighted(self) -> 'SparseGraph':
        """Convert to an unweighted graph (set all edge weights to 1).
        """
        self._adj_matrix.data = np.ones_like(self._adj_matrix.data)
        flag_writeable(self._adj_matrix, False)
        return self

    def is_connected(self) -> bool:
        """Check if the graph is connected.
        """
        return sp.csgraph.connected_components(self.adj_matrix, return_labels=False) == 1

    def has_self_loops(self) -> bool:
        """Check if the graph has self-loops.
        """
        return not np.allclose(self.adj_matrix.diagonal(), 0)

    def __repr__(self) -> str:
        props = []
        for prop_name in sparse_graph_properties:
            prop = getattr(self, prop_name)
            if prop is not None:
                if prop_name == 'metadata':
                    props.append(prop_name)
                else:
                    shape_string = 'x'.join([str(x) for x in prop.shape])
                    props.append("{} ({})".format(prop_name, shape_string))
        dir_string = 'Directed' if self.is_directed() else 'Undirected'
        weight_string = 'weighted' if self.is_weighted() else 'unweighted'
        conn_string = 'connected' if self.is_connected() else 'disconnected'
        loop_string = 'has self-loops' if self.has_self_loops() else 'no self-loops'
        return ("<{}, {} and {} SparseGraph with {} edges ({}). Data: {}>"
                .format(dir_string, weight_string, conn_string,
                        self.num_edges(warn=False), loop_string,
                        ', '.join(props)))

    # Quality of life (shortcuts)
    def standardize(
            self, make_unweighted: bool = True,
            make_undirected: bool = True,
            no_self_loops: bool = True,
            select_lcc: bool = True
            ) -> 'SparseGraph':
        """Perform common preprocessing steps: remove self-loops, make unweighted/undirected, select LCC.

        All changes are done inplace.

        Parameters
        ----------
        make_unweighted
            Whether to set all edge weights to 1.
        make_undirected
            Whether to make the adjacency matrix symmetric. Can only be used if make_unweighted is True.
        no_self_loops
            Whether to remove self loops.
        select_lcc
            Whether to select the largest connected component of the graph.

        """
        G = self
        if make_unweighted and G.is_weighted():
            G = G.to_unweighted()
        if make_undirected and G.is_directed():
            G = G.to_undirected()
        if no_self_loops and G.has_self_loops():
            G = remove_self_loops(G)
        if select_lcc and not G.is_connected():
            G = largest_connected_components(G, 1)
        self._adopt_graph(G)
        return G

    def unpack(self) -> Tuple[sp.csr_matrix,
                              Union[np.ndarray, sp.csr_matrix],
                              Union[np.ndarray, sp.csr_matrix],
                              np.ndarray]:
        """Return the (A, X, E, z) quadruplet.
        """
        return self.adj_matrix, self.attr_matrix, self.edge_attr_matrix, self.labels

    def _adopt_graph(self, graph: 'SparseGraph'):
        """Copy all properties from the given graph to this graph.
        """
        for prop in sparse_graph_properties:
            setattr(self, '_{}'.format(prop), getattr(graph, prop))
        self._flag_writeable(False)

    def _flag_writeable(self, writeable: bool):
        """Flag all Numpy arrays and sparse matrices as non-writeable.
        """
        flag_writeable(self._adj_matrix, writeable)
        flag_writeable(self._attr_matrix, writeable)
        flag_writeable(self._edge_attr_matrix, writeable)
        flag_writeable(self._labels, writeable)
        flag_writeable(self._node_names, writeable)
        flag_writeable(self._attr_names, writeable)
        flag_writeable(self._edge_attr_names, writeable)
        flag_writeable(self._class_names, writeable)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return flat dictionary containing all SparseGraph properties.
        """
        data_dict = {}
        for key in sparse_graph_properties:
            val = getattr(self, key)
            if sp.isspmatrix(val):
                data_dict['{}.data'.format(key)] = val.data
                data_dict['{}.indices'.format(key)] = val.indices
                data_dict['{}.indptr'.format(key)] = val.indptr
                data_dict['{}.shape'.format(key)] = val.shape
            else:
                data_dict[key] = val
        return data_dict

    @staticmethod
    def from_flat_dict(data_dict: Dict[str, Any]) -> 'SparseGraph':
        """Initialize SparseGraph from a flat dictionary.
        """
        init_dict = {}
        del_entries = []

        # Construct sparse matrices
        for key in data_dict.keys():
            if key.endswith('_data') or key.endswith('.data'):
                if key.endswith('_data'):
                    sep = '_'
                    warnings.warn(
                            "The separator used for sparse matrices during export (for .npz files) "
                            "is now '.' instead of '_'. Please update (re-save) your stored graphs.",
                            DeprecationWarning, stacklevel=2)
                else:
                    sep = '.'
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = '{}{}indices'.format(matrix_name, sep)
                mat_indptr = '{}{}indptr'.format(matrix_name, sep)
                mat_shape = '{}{}shape'.format(matrix_name, sep)
                if matrix_name == 'adj' or matrix_name == 'attr':
                    warnings.warn(
                            "Matrices are exported (for .npz files) with full names now. "
                            "Please update (re-save) your stored graphs.",
                            DeprecationWarning, stacklevel=2)
                    matrix_name += '_matrix'
                init_dict[matrix_name] = sp.csr_matrix(
                        (data_dict[mat_data],
                         data_dict[mat_indices],
                         data_dict[mat_indptr]),
                        shape=data_dict[mat_shape])
                del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])

        # Delete sparse matrix entries
        for del_entry in del_entries:
            del data_dict[del_entry]

        # Load everything else
        for key, val in data_dict.items():
            if ((val is not None) and (None not in val)):
                init_dict[key] = val

        # Check if the dictionary contains only entries in sparse_graph_properties
        unknown_keys = [key for key in init_dict.keys() if key not in sparse_graph_properties]
        if len(unknown_keys) > 0:
            raise ValueError("Input dictionary contains keys that are not SparseGraph properties ({})."
                             .format(unknown_keys))

        return SparseGraph(**init_dict)

    @property
    def adj_matrix(self) -> sp.csr_matrix:
        return self._adj_matrix

    @property
    def attr_matrix(self) -> Union[np.ndarray, sp.csr_matrix]:
        return self._attr_matrix

    @property
    def edge_attr_matrix(self) -> Union[np.ndarray, sp.csr_matrix]:
        return self._edge_attr_matrix

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def node_names(self) -> np.ndarray:
        return self._node_names

    @property
    def attr_names(self) -> np.ndarray:
        return self._attr_names

    @property
    def edge_attr_names(self) -> np.ndarray:
        return self._edge_attr_names

    @property
    def class_names(self) -> np.ndarray:
        return self._class_names

    @property
    def metadata(self) -> Any:
        return self._metadata


def flag_writeable(matrix: Union[np.ndarray, sp.csr_matrix], writeable: bool):
    if matrix is not None:
        if sp.isspmatrix(matrix):
            matrix.data.flags.writeable = writeable
            matrix.indices.flags.writeable = writeable
            matrix.indptr.flags.writeable = writeable
        elif isinstance(matrix, np.ndarray):
            matrix.flags.writeable = writeable
        else:
            raise ValueError(
                    "Matrix must be an sp.spmatrix or an np.ndarray"
                    " (got {0} instead).".format(type(matrix)))


def create_subgraph(
        sparse_graph: SparseGraph,
        _sentinel: None = None,
        nodes_to_remove: np.ndarray = None,
        nodes_to_keep: np.ndarray = None
        ) -> SparseGraph:
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    The subgraph partially points to the old graph's data.

    Parameters
    ----------
    sparse_graph
        Input graph.
    _sentinel
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove
        Indices of nodes that have to removed.
    nodes_to_keep
        Indices of nodes that have to be kept.

    Returns
    -------
    SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...).")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is None:
        attr_matrix = None
    else:
        attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.edge_attr_matrix is None:
        edge_attr_matrix = None
    else:
        old_idx = sparse_graph.get_edgeid_to_idx_array()
        keep_edge_idx = np.where(np.all(np.isin(old_idx, nodes_to_keep), axis=1))[0]
        edge_attr_matrix = sparse_graph.edge_attr_matrix[keep_edge_idx]
    if sparse_graph.labels is None:
        labels = None
    else:
        labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is None:
        node_names = None
    else:
        node_names = sparse_graph.node_names[nodes_to_keep]
    return SparseGraph(
            adj_matrix, attr_matrix, edge_attr_matrix, labels, node_names,
            sparse_graph.attr_names, sparse_graph.edge_attr_names,
            sparse_graph.class_names, sparse_graph.metadata)


def largest_connected_components(sparse_graph: SparseGraph, n_components: int = 1) -> SparseGraph:
    """Select the largest connected components in the graph.

    Changes are returned in a partially new SparseGraph.

    Parameters
    ----------
    sparse_graph
        Input graph.
    n_components
        Number of largest connected components to keep.

    Returns
    -------
    SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    # TODO: add warnings / logging
    # print("Selecting {0} largest connected components".format(n_components))
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def remove_self_loops(sparse_graph: SparseGraph) -> SparseGraph:
    """Remove self loops (diagonal entries in the adjacency matrix).

    Changes are returned in a partially new SparseGraph.

    """
    num_self_loops = (~np.isclose(sparse_graph.adj_matrix.diagonal(), 0)).sum()
    if num_self_loops > 0:
        adj_matrix = sparse_graph.adj_matrix.copy().tolil()
        adj_matrix.setdiag(0)
        adj_matrix = adj_matrix.tocsr()
        if sparse_graph.edge_attr_matrix is None:
            edge_attr_matrix = None
        else:
            old_idx = sparse_graph.get_edgeid_to_idx_array()
            keep_edge_idx = np.where((old_idx[:, 0] - old_idx[:, 1]) != 0)[0]
            edge_attr_matrix = sparse_graph._edge_attr_matrix[keep_edge_idx]
        warnings.warn("{0} self loops removed".format(num_self_loops))
        return SparseGraph(
                adj_matrix, sparse_graph.attr_matrix, edge_attr_matrix,
                sparse_graph.labels, sparse_graph.node_names,
                sparse_graph.attr_names, sparse_graph.edge_attr_names,
                sparse_graph.class_names, sparse_graph.metadata)
    else:
        return sparse_graph
