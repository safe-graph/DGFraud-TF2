"""
This code is attributed to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou),
Zhongzheng Lu(@lzz-hub-dev) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

Paper: 'Key Player Identification in Underground Forums
over Attributed Heterogeneous Information Network Embedding Framework'
Link: http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf
"""

import argparse
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from layers.layers import AttentionLayer, GraphConvolution
from utils.metrics import masked_softmax_cross_entropy, masked_accuracy


class Player2Vec(keras.Model):
    """
    The Player2Vec model
    """

    def __init__(self, input_dim: int, nhid: int, output_dim: int,
                 args: argparse.ArgumentParser().parse_args()) -> None:
        """
        :param input_dim: the input feature dimension
        :param nhid: the output embedding dimension of the first GCN layer
        :param output_dim: the output embedding dimension of the last GCN layer
        (number of classes)
        :param args: additional parameters
        """
        super().__init__()

        self.input_dim = input_dim
        self.nodes = args.nodes
        self.nhid = nhid
        self.class_size = args.class_size
        self.train_size = args.train_size
        self.output_dim = output_dim
        self.weight_decay = args.weight_decay
        self.num_features_nonzero = args.num_features_nonzero

        self.layers_ = []
        self.layers_.append(GraphConvolution(input_dim=self.input_dim,
                                             output_dim=self.nhid,
                                             num_features_nonzero=self.
                                             num_features_nonzero,
                                             activation=tf.nn.relu,
                                             dropout=args.dropout,
                                             is_sparse_inputs=True,
                                             norm=True))

        self.layers_.append(GraphConvolution(input_dim=self.nhid,
                                             output_dim=self.output_dim,
                                             num_features_nonzero=self.
                                             num_features_nonzero,
                                             activation=lambda x: x,
                                             dropout=args.dropout,
                                             norm=False))

    def call(self, inputs: list, training: bool = True) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward propagation
        :param inputs: the information passed to next layers
        :param training: whether in the training mode
        """
        supports, x, label, mask = inputs
        outputs = []

        # forward propagation
        for i in range(len(supports)):
            output = [x]
            for layer in self.layers:
                hidden = layer((output[-1], [supports[i]]), training)
                output.append(hidden)
            output = output[-1]
            outputs.append(output)
        outputs = tf.reshape(outputs,
                             [len(supports), self.nodes * self.output_dim])
        outputs = AttentionLayer.attention(inputs=outputs,
                                           attention_size=len(supports),
                                           v_type='tanh')
        outputs = tf.reshape(outputs, [self.nodes, self.output_dim])

        # Weight decay loss
        loss = tf.zeros([])
        for var in self.layers_[0].trainable_variables:
            loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy loss
        loss += masked_softmax_cross_entropy(outputs, label, mask)

        # Prediction results
        acc = masked_accuracy(outputs, label, mask)

        return loss, acc
