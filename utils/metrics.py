"""
DISCLAIMER:
Parts of this code file were originally forked from
https://github.com/tkipf/gcn
"""

import tensorflow as tf


def masked_softmax_cross_entropy(preds: tf.Tensor, labels: tf.Tensor,
                                 mask: tf.Tensor) -> tf.Tensor:
    """
    Softmax cross-entropy loss with masking.

    :param preds: the last layer logits of the input data
    :param labels: the labels of the input data
    :param mask: the mask for train/val/test data
    """
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.maximum(tf.reduce_sum(mask), tf.constant([1.]))
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds: tf.Tensor, labels: tf.Tensor,
                    mask: tf.Tensor) -> tf.Tensor:
    """
    Accuracy with masking.

    :param preds: the class prediction probabilities of the input data
    :param labels: the labels of the input data
    :param mask: the mask for train/val/test data
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def accuracy(preds: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Accuracy.

    :param preds: the class prediction probabilities of the input data
    :param labels: the labels of the input data
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_sum(accuracy_all) / preds.shape[0]
