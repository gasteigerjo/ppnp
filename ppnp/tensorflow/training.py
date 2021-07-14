from typing import Type
import time
import os
import logging
import tensorflow as tf

from ..data.sparsegraph import SparseGraph
from .model import Model
from ..preprocessing import gen_splits, gen_seeds
from .earlystopping import EarlyStopping, stopping_args


def train_model(
        name: str, model_class: Type[Model], graph: SparseGraph, build_args: dict,
        idx_split_args: dict = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114},
        stopping_args: dict = stopping_args,
        test: bool = False, save_result: bool = False,
        tf_seed: int = None, print_interval: int = 20) -> dict:
    labels = graph.labels
    train_idx, stopping_idx, valtest_idx = gen_splits(
            labels, idx_split_args, test=test)

    logging.log(21, f"{model_class.__name__}: {build_args}")
    tf.reset_default_graph()
    if tf_seed is None:
        tf_seed = gen_seeds()
    tf.set_random_seed(tf_seed)
    logging.log(22, f"Tensorflow seed: {tf_seed}")
    sess = tf.Session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    model = model_class(graph.attr_matrix, labels, sess)
    model.build_model(**build_args)

    train_inputs = {
            model.idx: train_idx,
            model.isTrain: True}
    train_inference_inputs = {
            model.idx: train_idx,
            model.isTrain: False}
    stopping_inputs = {
            model.idx: stopping_idx,
            model.isTrain: False}
    valtest_inputs = {
            model.idx: valtest_idx,
            model.isTrain: False}

    init = tf.global_variables_initializer()
    sess.run(init)

    if save_result:
        log_dir = '..\\{}_{}'.format(model_class.__name__, name)
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        checkpoint_saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(
                os.path.join(log_dir, 'train'),
                sess.graph)
        stopping_writer = tf.summary.FileWriter(
                os.path.join(log_dir, 'early_stopping'),
                sess.graph)
        valtest_writer = tf.summary.FileWriter(
                os.path.join(log_dir, 'valtest'),
                sess.graph)

    early_stopping = EarlyStopping(model, **stopping_args)

    start_time = time.time()
    last_time = start_time
    for step in range(early_stopping.max_steps):

        _, train_loss = sess.run(
                [model.train_op, model.loss],
                feed_dict=train_inputs)

        train_acc, train_str = sess.run(
                [model.accuracy, model.summary], feed_dict=train_inference_inputs)
        if save_result:
            train_writer.add_summary(train_str, step)
        stopping_loss, stopping_acc, stopping_str = sess.run(
                [model.loss, model.accuracy, model.summary],
                feed_dict=stopping_inputs)
        if save_result:
            stopping_writer.add_summary(stopping_str, step)
        valtest_str = sess.run(model.summary, feed_dict=valtest_inputs)
        if save_result:
            valtest_writer.add_summary(valtest_str, step)

        if step % print_interval == 0:
            duration = time.time() - last_time
            last_time = time.time()
            logging.info(
                    "Step {}: Train loss = {:.2f}, train acc = {:.1f}, "
                    "early stopping loss = {:.2f}, early stopping acc = {:.1f} ({:.3f} sec)"
                    .format(step, train_loss, train_acc * 100,
                            stopping_loss, stopping_acc * 100, duration))
        if len(early_stopping.stop_vars) > 0:
            stop_vars = sess.run(
                    early_stopping.stop_vars, feed_dict=stopping_inputs)
            if early_stopping.check(stop_vars, step):
                break
    runtime = time.time() - start_time
    runtime_perepoch = runtime / (step + 1)

    if len(early_stopping.stop_vars) == 0:
        logging.log(22, "Last step: {} ({:.3f} sec)".format(step, runtime))
    else:
        logging.log(22, "Last step: {}, best step: {} ({:.3f} sec)"
                    .format(step, early_stopping.best_step, runtime))
        model.set_vars(early_stopping.best_trainables)

    train_accuracy, train_f1_score = sess.run(
            [model.accuracy, model.f1_score],
            feed_dict=train_inputs)

    stopping_accuracy, stopping_f1_score = sess.run(
            [model.accuracy, model.f1_score],
            feed_dict=stopping_inputs)
    logging.log(21, "Early stopping accuracy: {:.1f}%, early stopping F1 score: {:.3f}"
                .format(stopping_accuracy * 100, stopping_f1_score))

    valtest_accuracy, valtest_f1_score = sess.run(
            [model.accuracy, model.f1_score],
            feed_dict=valtest_inputs)

    valtest_name = 'Test' if test else 'Validation'
    logging.log(22, "{} accuracy: {:.1f}%, test F1 score: {:.3f}"
                .format(valtest_name, valtest_accuracy * 100, valtest_f1_score))

    if save_result:
        if len(early_stopping.stop_vars) == 0:
            checkpoint_saver.save(sess, checkpoint_file, global_step=step)
        else:
            checkpoint_saver.save(
                    sess, checkpoint_file,
                    global_step=early_stopping.best_step)
        train_writer.flush()
        stopping_writer.flush()
        valtest_writer.flush()

    result = {}
    result['predictions'] = model.get_predictions()
    result['vars'] = early_stopping.best_trainables
    result['train'] = {'accuracy': train_accuracy, 'f1_score': train_f1_score}
    result['early_stopping'] = {'accuracy': stopping_accuracy, 'f1_score': stopping_f1_score}
    result['valtest'] = {'accuracy': valtest_accuracy, 'f1_score': valtest_f1_score}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime_perepoch
    sess.close()
    return result
