"""
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou),
Zhongzheng Lu(@lzz-hub-dev) and UIC BDSC Lab
DGFraud-TF2 (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2

Paper: 'A Semi-supervised Graph Attentive Network for
        Financial Fraud Detection'
Link: https://arxiv.org/pdf/2003.01171
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from layers.layers import NodeAttention, ViewAttention
from utils.metrics import accuracy


class SemiGNN(keras.Model):
    """
    The SemiGNN model
    """
    def __init__(self, nodes: int, class_size: int, semi_encoding1: int,
                 semi_encoding2: int, semi_encoding3: int, init_emb_size: int,
                 view_num: int, alpha: float) -> None:
        """
        :param nodes: total nodes number
        :param semi_encoding1: the first view attention layer unit number
        :param semi_encoding2: the second view attention layer unit number
        :param semi_encoding3: MLP layer unit number
        :param init_emb_size: the initial node embedding
        :param view_num: number of views
        :param alpha: the coefficient of loss function
        """
        super().__init__()

        self.nodes = nodes
        self.class_size = class_size
        self.semi_encoding1 = semi_encoding1
        self.semi_encoding2 = semi_encoding2
        self.semi_encoding3 = semi_encoding3
        self.init_emb_size = init_emb_size
        self.view_num = view_num
        self.alpha = alpha

        # init embedding
        self.x_init = tf.keras.initializers.GlorotUniform()
        self.emb = tf.Variable(
            initial_value=self.x_init(shape=(self.nodes, self.init_emb_size),
                                      dtype=tf.float32),
            trainable=True)

        self.node_att_layer = []
        for _ in range(view_num):
            self.node_att_layer.append(NodeAttention(input_dim=init_emb_size))

        # we define a two layer MLP for Eq. (2) in the paper
        encoding = [self.semi_encoding1, self.semi_encoding2]
        self.view_att_layer = ViewAttention(layer_size=len(encoding),
                                            view_num=self.view_num,
                                            encoding=encoding)

        # the one-layer perceptron used to refine the embedding
        self.olp = tf.keras.layers.Dense(self.semi_encoding3)

        # the parameter for softmax for Eq. (5)
        self.theta = tf.Variable(initial_value=self.x_init(
            shape=(self.semi_encoding3, self.class_size), dtype=tf.float32),
            trainable=True)

    def call(self, inputs: list, training: bool = True) -> \
            Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward propagation
        :param inputs: the information passed to next layers
        :param training: whether in the training mode
        """
        adj_data, u_i, u_j, graph_label, label, idx_mask = inputs

        # node level attention
        h1 = []
        for v in range(self.view_num):
            h = self.node_att_layer[v]([self.emb, adj_data[v]])
            h = tf.reshape(h, [self.nodes, self.emb.shape[1]])
            h1.append(h)
        h1 = tf.concat(h1, 0)
        h1 = tf.reshape(h1, [self.view_num, self.nodes, self.init_emb_size])

        # view level attention
        h2 = self.view_att_layer(h1)
        a_u = self.olp(h2)

        # get masked data
        masked_data = tf.gather(a_u, idx_mask)
        masked_label = tf.gather(label, idx_mask)

        # calculation loss and accuracy
        logits = tf.nn.softmax(tf.matmul(masked_data, self.theta))

        # Eq. (5)
        loss1 = -(1 / len(idx_mask)) * tf.reduce_sum(
            masked_label * tf.math.log(tf.nn.softmax(logits)))

        u_i_embedding = tf.nn.embedding_lookup(a_u,
                                               tf.cast(u_i, dtype=tf.int32))
        u_j_embedding = tf.nn.embedding_lookup(a_u,
                                               tf.cast(u_j, dtype=tf.int32))
        inner_product = tf.reduce_sum(u_i_embedding * u_j_embedding, axis=1)

        # Eq. (6)
        loss2 = -tf.reduce_mean(
            tf.math.log_sigmoid(graph_label * inner_product))

        # Eq. (7)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        acc = accuracy(logits, masked_label)

        return loss, acc
