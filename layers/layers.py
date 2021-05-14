"""
This code is attributed to Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import tensorflow as tf
from typing import Callable, Optional, Tuple
from tensorflow.keras import layers


def sparse_dropout(x: tf.SparseTensor, rate: float,
                   noise_shape: int) -> tf.SparseTensor:
    """
    Dropout for sparse tensors.

    :param x: the input sparse tensor
    :param rate: the dropout rate
    :param noise_shape: the feature dimension
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / (1 - rate))


def dot(x: tf.Tensor, y: tf.Tensor, sparse: bool = False) -> tf.Tensor:
    """
    Wrapper for tf.matmul (sparse vs dense).

    :param x: first tensor
    :param y: second tensor
    :param sparse: whether the first tensor is of type tf.SparseTensor
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    Source:https://github.com/dragen1860/GCN-TF2/blob/master/layers.py

    :param input_dim: the input feature dimension
    :param output_dim: the output dimension (number of classes)
    :param num_features_nonzero: the node feature dimension
    :param dropout: the dropout rate
    :param is_sparse_inputs: whether the input feature/adj are sparse matrices
    :param activation: the activation function
    :param norm: whether adding L2-normalization to parameters
    :param bias: whether adding bias term to the output
    :param featureless: whether the input has features
    """

    def __init__(self, input_dim: int, output_dim: int,
                 num_features_nonzero: int,
                 dropout: float = 0.,
                 is_sparse_inputs: bool = False,
                 activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 norm: bool = False,
                 bias: bool = False,
                 featureless: bool = False, **kwargs: Optional) -> None:
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.norm = norm
        self.num_features_nonzero = num_features_nonzero

        self.weights_ = []
        for i in range(1):
            w = self.add_variable('weight' + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_variable('bias', [output_dim])

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: bool = True) -> tf.Tensor:
        """
        Forward propagation

        :param inputs: the information passed to next layers
        :param training: whether in the training mode
        """
        x, support_ = inputs

        # dropout
        if training is not False and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)

        # convolve
        supports = list()
        for i in range(len(support_)):
            if not self.featureless:  # if it has features x
                pre_sup = dot(x, self.weights_[i],
                              sparse=self.is_sparse_inputs)
            else:
                pre_sup = self.weights_[i]

            support = dot(support_[i], pre_sup, sparse=True)
            supports.append(support)

        output = tf.add_n(supports)
        axis = list(range(len(output.get_shape()) - 1))
        mean, variance = tf.nn.moments(output, axis)
        scale = None
        offset = None
        variance_epsilon = 0.001
        output = tf.nn.batch_normalization(output, mean, variance, offset,
                                           scale, variance_epsilon)

        # bias
        if self.bias:
            output += self.bias
        if self.norm:
            return tf.nn.l2_normalize(self.activation(output), axis=None,
                                      epsilon=1e-12)

        return self.activation(output)
