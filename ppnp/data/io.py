from numbers import Number
from typing import Union
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from .sparsegraph import SparseGraph

data_dir = Path(__file__).parent


def load_from_npz(file_name: str) -> SparseGraph:
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name
        Name of the file to load.

    Returns
    -------
    SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        dataset = SparseGraph.from_flat_dict(loader)
    return dataset


def load_dataset(name: str,
                 directory: Union[Path, str] = data_dir
                 ) -> SparseGraph:
    """Load a dataset.

    Parameters
    ----------
    name
        Name of the dataset to load.
    directory
        Path to the directory where the datasets are stored.

    Returns
    -------
    SparseGraph
        The requested dataset in sparse format.

    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not name.endswith('.npz'):
        name += '.npz'
    path_to_file = directory / name
    if path_to_file.exists():
        return load_from_npz(path_to_file)
    else:
        raise ValueError("{} doesn't exist.".format(path_to_file))


def networkx_to_sparsegraph(
        nx_graph: Union['nx.Graph', 'nx.DiGraph'],
        label_name: str = None,
        sparse_node_attrs: bool = True,
        sparse_edge_attrs: bool = True
        ) -> 'SparseGraph':
    """Convert NetworkX graph to SparseGraph.

    Node attributes need to be numeric.
    Missing entries are interpreted as 0.
    Labels can be any object. If non-numeric they are interpreted as
    categorical and enumerated.

    This ignores all edge attributes except the edge weights.

    Parameters
    ----------
    nx_graph
        Graph to convert.

    Returns
    -------
    SparseGraph
        Converted graph.

    """
    import networkx as nx

    # Extract node names
    int_names = True
    for node in nx_graph.nodes:
        int_names &= isinstance(node, int)
    if int_names:
        node_names = None
    else:
        node_names = np.array(nx_graph.nodes)
        nx_graph = nx.convert_node_labels_to_integers(nx_graph)

    # Extract adjacency matrix
    adj_matrix = nx.adjacency_matrix(nx_graph)

    # Collect all node attribute names
    attrs = set()
    for _, node_data in nx_graph.nodes().data():
        attrs.update(node_data.keys())

    # Initialize labels and remove them from the attribute names
    if label_name is None:
        labels = None
    else:
        if label_name not in attrs:
            raise ValueError("No attribute with label name '{}' found.".format(label_name))
        attrs.remove(label_name)
        labels = [0 for _ in range(nx_graph.number_of_nodes())]

    if len(attrs) > 0:
        # Save attribute names if not integer
        all_integer = all((isinstance(attr, int) for attr in attrs))
        if all_integer:
            attr_names = None
            attr_mapping = None
        else:
            attr_names = np.array(list(attrs))
            attr_mapping = {k: i for i, k in enumerate(attr_names)}

        # Initialize attribute matrix
        if sparse_node_attrs:
            attr_matrix = sp.lil_matrix((nx_graph.number_of_nodes(), len(attr_names)), dtype=np.float32)
        else:
            attr_matrix = np.zeros((nx_graph.number_of_nodes(), len(attr_names)), dtype=np.float32)
    else:
        attr_matrix = None
        attr_names = None

    # Fill label and attribute matrices
    for inode, node_attrs in nx_graph.nodes.data():
        for key, val in node_attrs.items():
            if key == label_name:
                labels[inode] = val
            else:
                if not isinstance(val, Number):
                    if node_names is None:
                        raise ValueError("Node {} has attribute '{}' with value '{}', which is not a number."
                                         .format(inode, key, val))
                    else:
                        raise ValueError("Node '{}' has attribute '{}' with value '{}', which is not a number."
                                         .format(node_names[inode], key, val))
                if attr_mapping is None:
                    attr_matrix[inode, key] = val
                else:
                    attr_matrix[inode, attr_mapping[key]] = val
    if attr_matrix is not None and sparse_node_attrs:
        attr_matrix = attr_matrix.tocsr()

    # Convert labels to integers
    if labels is None:
        class_names = None
    else:
        try:
            labels = np.array(labels, dtype=np.float32)
            class_names = None
        except ValueError:
            class_names = np.unique(labels)
            class_mapping = {k: i for i, k in enumerate(class_names)}
            labels_int = np.empty(nx_graph.number_of_nodes(), dtype=np.float32)
            for inode, label in enumerate(labels):
                labels_int[inode] = class_mapping[label]
            labels = labels_int

    return SparseGraph(
            adj_matrix=adj_matrix, attr_matrix=attr_matrix, labels=labels,
            node_names=node_names, attr_names=attr_names, class_names=class_names,
            metadata=None)
