"""
This code is attributed to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou),
Zhongzheng Lu(@lzz-hub-dev) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

Paper: 'Spam Review Detection with Graph Convolutional Networks'
Link: https://arxiv.org/abs/1908.10679
"""

import argparse
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from layers.layers import ConcatenationAggregator, AttentionAggregator, \
    GASConcatenation, GraphConvolution
from utils.metrics import accuracy


class GAS(keras.Model):
    """
    The GAS model
    """

    def __init__(self, args: argparse.ArgumentParser().parse_args()) -> None:
        """
        :param args: argument used by the GAS model
        """
        super().__init__()

        self.class_size = args.class_size
        self.reviews_num = args.reviews_num
        self.input_dim_i = args.input_dim_i
        self.input_dim_u = args.input_dim_u
        self.input_dim_r = args.input_dim_r
        self.input_dim_r_gcn = args.input_dim_r_gcn
        self.output_dim1 = args.output_dim1
        self.output_dim2 = args.output_dim2
        self.output_dim3 = args.output_dim3
        self.output_dim4 = args.output_dim4
        self.output_dim5 = args.output_dim5
        self.num_features_nonzero = args.num_features_nonzero
        self.gcn_dim = args.gcn_dim
        self.h_i_size = args.h_i_size
        self.h_u_size = args.h_u_size
        self.input_dim_u_x = args.input_dim_u_x
        self.input_dim_i_x = args.input_dim_i_x

        # GAS layers initialization
        self.r_agg_layer = ConcatenationAggregator(
            input_dim=self.input_dim_r + self.input_dim_u + self.input_dim_i,
            output_dim=self.output_dim1, )

        # item user aggregator
        self.iu_agg_layer = AttentionAggregator(input_dim1=self.h_u_size,
                                                input_dim2=self.h_i_size,
                                                output_dim=self.output_dim3,
                                                hid_dim=self.output_dim2,
                                                input_dim_u_x=self.input_dim_u_x,
                                                input_dim_i_x=self.input_dim_i_x,
                                                concat=True)

        # review aggregator
        self.r_gcn_layer = GraphConvolution(input_dim=self.input_dim_r_gcn,
                                            output_dim=self.output_dim5,
                                            num_features_nonzero=self.
                                            num_features_nonzero,
                                            activation=tf.nn.relu,
                                            dropout=args.dropout,
                                            is_sparse_inputs=True,
                                            norm=True)

        self.concat_layer = GASConcatenation()

        # logistic weights initialization
        self.x_init = tf.keras.initializers.GlorotUniform()
        self.u = tf.Variable(initial_value=self.x_init(
            shape=(
                self.output_dim1 + 2 * self.output_dim2 + self.input_dim_i
                + self.input_dim_u + self.output_dim5,
                self.class_size),
            dtype=tf.float32), trainable=True)

    def call(self, inputs: list, training: bool = True) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward propagation
        :param inputs: the information passed to next layers
        :param training: whether in the training mode
        """
        support, r_support, features, r_feature, label, idx_mask = inputs

        # forward propagation
        h_r = self.r_agg_layer((support, features))
        h_u, h_i = self.iu_agg_layer((support, features))
        p_e = self.r_gcn_layer((r_feature, r_support), training=True)
        concat_vecs = [h_r, h_u, h_i, p_e]
        gas_out = self.concat_layer((support, concat_vecs))

        # get masked data
        masked_data = tf.gather(gas_out, idx_mask)
        masked_label = tf.gather(label, idx_mask)

        # calculation loss and accuracy()
        logits = tf.nn.softmax(tf.matmul(masked_data, self.u))
        loss = -tf.reduce_sum(
            tf.math.log(tf.nn.sigmoid(masked_label * logits)))
        acc = accuracy(logits, masked_label)

        return loss, acc
