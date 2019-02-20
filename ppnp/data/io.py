from typing import Union
from pathlib import Path
import numpy as np

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
