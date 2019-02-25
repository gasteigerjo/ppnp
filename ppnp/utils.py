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


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    after_exp = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return after_exp / np.sum(after_exp, axis=axis, keepdims=True)


def get_accuracy(predictions: np.ndarray, labels: np.ndarray) -> np.floating:
    return np.mean(predictions == labels)


def get_f1(predictions: np.ndarray, labels: np.ndarray) -> np.floating:
    nclasses = np.max(labels) + 1
    avg_precision = 0
    avg_recall = 0
    for i in range(nclasses):
        pred_is_i = predictions == i
        label_is_i = labels == i
        true_pos = np.sum(pred_is_i & label_is_i)
        false_pos = np.sum(pred_is_i & ~label_is_i)
        false_neg = np.sum(~pred_is_i & label_is_i)
        if false_pos == 0:
            avg_precision += 1.
        else:
            avg_precision += true_pos / (true_pos + false_pos)
        if false_neg == 0:
            avg_recall += 1.
        else:
            avg_recall += true_pos / (true_pos + false_neg)
    avg_precision /= nclasses
    avg_recall /= nclasses
    f1_score = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall))
    return f1_score
