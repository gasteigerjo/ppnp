from typing import Union
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def sparse_matrix_to_tensor(X: sp.spmatrix) -> tf.SparseTensor:
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(
            indices,
            np.array(coo.data, dtype=np.float32),
            coo.shape)


def matrix_to_tensor(
        X: Union[np.ndarray, sp.spmatrix]) -> Union[tf.Tensor, tf.SparseTensor]:
    if sp.issparse(X):
        return sparse_matrix_to_tensor(X)
    else:
        return tf.constant(X, dtype=tf.float32)


def sparse_dropout(X: tf.SparseTensor, keep_prob: float) -> tf.SparseTensor:
    X_drop_val = tf.nn.dropout(X.values, keep_prob)
    return tf.SparseTensor(X.indices, X_drop_val, X.dense_shape)


def mixed_dropout(
        X: Union[tf.Tensor, tf.SparseTensor],
        keep_prob: float) -> Union[tf.Tensor, tf.SparseTensor]:
    if isinstance(X, tf.SparseTensor):
        return sparse_dropout(X, keep_prob)
    else:
        return tf.nn.dropout(X, keep_prob)
