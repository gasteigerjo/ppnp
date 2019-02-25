from typing import Union
import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.sparse as sp

from .utils import matrix_to_tensor


class Model:
    def __init__(
            self, attr_matrix: Union[np.ndarray, sp.spmatrix],
            labels: np.ndarray, sess: tf.Session):
        self.attr_matrix = matrix_to_tensor(attr_matrix)
        self.labels_np = labels
        self.labels = tf.constant(labels, dtype=tf.int32)
        self.sess = sess
        self.nnodes, self.nfeatures = attr_matrix.shape
        self.nclasses = np.max(labels) + 1
        self.reg_vars = []
        self.step = tf.train.create_global_step()

    def _calc_cross_entropy(self):
        logits_subset = tf.gather(self.logits, self.idx, axis=0)
        labels_subset = tf.gather(self.labels, self.idx, axis=0)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_subset, logits=logits_subset,
                name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
        return cross_entropy_mean

    def _calc_f1_accuracy(self, predictions):
        predictions_subset = tf.gather(predictions, self.idx, axis=0)
        labels_subset = tf.gather(self.labels, self.idx, axis=0)
        with tf.variable_scope('Accuracy'):
            correct = tf.equal(predictions_subset, labels_subset)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('Accuracy', accuracy)
        with tf.variable_scope('F1_score'):
            def sum_cast(x, dtype=tf.float32):
                return tf.reduce_sum(tf.cast(x, dtype))
            for i in range(self.nclasses):
                pred_is_i = tf.equal(predictions_subset, tf.constant(i))
                label_is_i = tf.equal(labels_subset, tf.constant(i))
                true_pos = sum_cast(pred_is_i & label_is_i)
                false_pos = sum_cast(pred_is_i & ~label_is_i)
                false_neg = sum_cast(~pred_is_i & label_is_i)
                precision = tf.cond(
                        tf.equal(false_pos, 0),
                        lambda: 1.,
                        lambda: true_pos / (true_pos + false_pos))
                recall = tf.cond(
                        tf.equal(false_neg, 0),
                        lambda: 1.,
                        lambda: true_pos / (true_pos + false_neg))
                if i == 0:
                    avg_precision = precision
                    avg_recall = recall
                else:
                    avg_precision += precision
                    avg_recall += recall
            avg_precision /= self.nclasses
            avg_recall /= self.nclasses
            f1_score = (2 * (avg_precision * avg_recall)
                        / (avg_precision + avg_recall))
            tf.summary.scalar('F1_score', f1_score)
        return f1_score, accuracy

    def _build_loss(self, reg_lambda):
        with tf.variable_scope('Loss'):
            cross_entropy_mean = self._calc_cross_entropy()
            l2_reg = tf.add_n([
                    tf.nn.l2_loss(weight) for weight in self.reg_vars])
            self.loss = cross_entropy_mean + reg_lambda * l2_reg
            tf.summary.scalar('l2_reg', l2_reg)
            tf.summary.scalar('loss', self.loss)

    def _build_training(self, learning_rate):
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(
                loss=self.loss, global_step=tf.train.get_global_step())

    def _build_results(self):
        # Predictions, accuracy and F1-score
        self.predictions = tf.argmax(
                self.logits, axis=1, output_type=tf.int32)
        self.f1_score, self.accuracy = self._calc_f1_accuracy(
                self.predictions)
        self.summary = tf.summary.merge_all()

    def get_vars(self):
        return self.sess.run(tf.trainable_variables())

    def set_vars(self, new_vars):
        set_all = [
                var.assign(new_vars[i])
                for i, var in enumerate(tf.trainable_variables())]
        self.sess.run(set_all)

    def calc_confusion_matrix(self, idx):
        inputs = {
                self.idx: idx,
                self.isTrain: False}
        curr_predictions = self.sess.run(self.predictions, feed_dict=inputs)
        return pd.crosstab(
                curr_predictions[idx], self.labels_np[idx],
                rownames=['Predictions'], colnames=['Labels'])

    def get_logits(self):
        inputs = {
                self.idx: np.arange(self.nnodes),
                self.isTrain: False}
        return self.sess.run(self.logits, feed_dict=inputs)

    def get_predictions(self):
        inputs = {
                self.idx: np.arange(self.nnodes),
                self.isTrain: False}
        return self.sess.run(self.predictions, feed_dict=inputs)

    def get_hidden_activations(self):
        if hasattr(self, 'Zs'):
            inputs = {
                    self.idx: np.arange(self.nnodes),
                    self.isTrain: False}
            return self.sess.run(self.Zs, feed_dict=inputs)

    def update_train_op(self, step):
        pass
