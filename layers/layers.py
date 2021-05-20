"""
This code is attributed to Yingtong Dou (@YingtongDou),
Zhongzheng Lu(@lzz-hub-dev) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import tensorflow as tf
from typing import Callable, Optional, Tuple
from tensorflow.keras import layers

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

'''Code about GCN is adapted from tkipf/gcn.'''


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


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


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs,\
                'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def call(self, inputs, adj_info):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


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


class AttentionLayer(layers.Layer):
    """ AttentionLayer is a function f : hkey × Hval → hval which maps
    a feature vector hkey and the set of candidates’ feature vectors
    Hval to an weighted sum of elements in Hval.
    """

    def attention(inputs, attention_size, v_type=None, return_weights=False,
                  bias=True, joint_type='weighted_sum',
                  multi_view=True):
        if multi_view:
            inputs = tf.expand_dims(inputs, 0)
        hidden_size = inputs.shape[-1]

        # Trainable parameters
        w_omega = tf.Variable(tf.random.uniform([hidden_size, attention_size]))
        b_omega = tf.Variable(tf.random.uniform([attention_size]))
        u_omega = tf.Variable(tf.random.uniform([attention_size]))

        v = tf.tensordot(inputs, w_omega, axes=1)
        if bias is True:
            v += b_omega
        if v_type == 'tanh':
            v = tf.tanh(v)
        if v_type == 'relu':
            v = tf.nn.relu(v)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        weights = tf.nn.softmax(vu, name='alphas')

        if joint_type == 'weighted_sum':
            output = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), 1)
        if joint_type == 'concatenation':
            output = tf.concat(inputs * tf.expand_dims(weights, -1), 2)

        if not return_weights:
            return output
        else:
            return output, weights

    def node_attention(inputs, adj, return_weights=False):
        hidden_size = inputs.shape[-1]
        H_v = tf.Variable(tf.random.normal([hidden_size, 1], stddev=0.1))

        # convert adj to sparse tensor
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(adj, zero)
        indices = tf.where(where)
        values = tf.gather_nd(adj, indices)
        adj = tf.SparseTensor(indices=indices,
                              values=values,
                              dense_shape=adj.shape)
        v = tf.cast(adj, tf.float32) * tf.squeeze(
            tf.tensordot(inputs, H_v, axes=1))

        weights = tf.sparse.softmax(v, name='alphas')  # [nodes,nodes]
        output = tf.sparse.sparse_dense_matmul(weights, inputs)

        if not return_weights:
            return output
        else:
            return output, weights

    # view-level attention (equation (4) in SemiGNN)
    def view_attention(inputs, encoding1, encoding2, layer_size, meta,
                       return_weights=False):
        h = inputs
        encoding = [encoding1, encoding2]
        for j in range(layer_size):
            v = []
            for i in range(meta):
                input = h[i]
                v_i = tf.keras.layers.Dense(encoding[j])(input)
                v.append(v_i)
            h = v
        h = tf.concat(h, 0)
        h = tf.reshape(h, [meta, inputs[0].shape[0], encoding2])
        phi = tf.Variable(tf.random.normal([encoding2, ], stddev=0.1))
        weights = tf.nn.softmax(h * phi, name='alphas')
        output = tf.reshape(h * weights,
                            [1, inputs[0].shape[0] * encoding2 * meta])

        if not return_weights:
            return output
        else:
            return output, weights

    def scaled_dot_product_attention(q, k, v, mask):
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention += 1
        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(weights, v)
        return output, weights


class ConcatenationAggregator(layers.Layer):
    """This layer equals to the equation (3) in
    paper 'Spam Review Detection with Graph Convolutional Networks.'
    """

    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu,
                 concat=False, **kwargs):
        super(ConcatenationAggregator, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.concat = concat
        self.con_agg_weights = self.add_weight('con_agg_weights',
                                               [input_dim, output_dim],
                                               dtype=tf.float32)

    def __call__(self, inputs):
        adj_list, features = inputs

        review_vecs = tf.nn.dropout(features[0], self.dropout)
        user_vecs = tf.nn.dropout(features[1], self.dropout)
        item_vecs = tf.nn.dropout(features[2], self.dropout)

        # neighbor sample
        ri = tf.nn.embedding_lookup(item_vecs,
                                    tf.cast(adj_list[5], dtype=tf.int32))
        ri = tf.transpose(tf.random.shuffle(tf.transpose(ri)))

        ru = tf.nn.embedding_lookup(user_vecs,
                                    tf.cast(adj_list[4], dtype=tf.int32))
        ru = tf.transpose(tf.random.shuffle(tf.transpose(ru)))

        concate_vecs = tf.concat([review_vecs, ru, ri], axis=1)

        # [nodes] x [out_dim]
        output = dot(concate_vecs, self.con_agg_weights, sparse=False)

        return self.act(output)


class AttentionAggregator(layers.Layer):
    """This layer equals to equation (5) and equation (8) in
    paper 'Spam Review Detection with Graph Convolutional Networks.'
    """

    def __init__(self, input_dim1, input_dim2, output_dim, hid_dim,
                 dropout=0., bias=False, act=tf.nn.relu,
                 concat=False, **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        self.user_weights = self.add_weight('user_weights',
                                            [input_dim1, hid_dim],
                                            dtype=tf.float32)
        self.item_weights = self.add_weight('item_weights',
                                            [input_dim2, hid_dim],
                                            dtype=tf.float32)
        self.concate_user_weights = self.add_weight('concate_user_weights',
                                                    [hid_dim, output_dim],
                                                    dtype=tf.float32)
        self.concate_item_weights = self.add_weight('concate_item_weights',
                                                    [hid_dim, output_dim],
                                                    dtype=tf.float32)

        if self.bias:
            self.user_bias = self.add_weight('user_bias', [self.output_dim],
                                             dtype=tf.float32)
            self.item_bias = self.add_weight('item_bias', [self.output_dim],
                                             dtype=tf.float32)

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim

    def __call__(self, inputs):
        adj_list, features = inputs

        review_vecs = tf.nn.dropout(features[0], self.dropout)
        user_vecs = tf.nn.dropout(features[1], self.dropout)
        item_vecs = tf.nn.dropout(features[2], self.dropout)

        # num_samples = self.adj_info[4]

        # neighbor sample
        ur = tf.nn.embedding_lookup(review_vecs,
                                    tf.cast(adj_list[0], dtype=tf.int32))
        ur = tf.transpose(tf.random.shuffle(tf.transpose(ur)))
        # ur = tf.slice(ur, [0, 0], [-1, num_samples])

        ri = tf.nn.embedding_lookup(item_vecs,
                                    tf.cast(adj_list[1], dtype=tf.int32))
        ri = tf.transpose(tf.random.shuffle(tf.transpose(ri)))
        # ri = tf.slice(ri, [0, 0], [-1, num_samples])

        ir = tf.nn.embedding_lookup(review_vecs,
                                    tf.cast(adj_list[2], dtype=tf.int32))
        ir = tf.transpose(tf.random.shuffle(tf.transpose(ir)))
        # ir = tf.slice(ir, [0, 0], [-1, num_samples])

        ru = tf.nn.embedding_lookup(user_vecs,
                                    tf.cast(adj_list[3], dtype=tf.int32))
        ru = tf.transpose(tf.random.shuffle(tf.transpose(ru)))
        # ru = tf.slice(ru, [0, 0], [-1, num_samples])

        concate_user_vecs = tf.concat([ur, ri], axis=2)
        concate_item_vecs = tf.concat([ir, ru], axis=2)

        # concate neighbor's embedding
        s1 = tf.shape(concate_user_vecs)
        s2 = tf.shape(concate_item_vecs)
        concate_user_vecs = tf.reshape(concate_user_vecs,
                                       [s1[0], s1[1] * s1[2]])
        concate_item_vecs = tf.reshape(concate_item_vecs,
                                       [s2[0], s2[1] * s2[2]])

        # attention
        concate_user_vecs, _ = AttentionLayer.scaled_dot_product_attention(
            q=user_vecs, k=user_vecs,
            v=concate_user_vecs,
            mask=None)
        concate_item_vecs, _ = AttentionLayer.scaled_dot_product_attention(
            q=item_vecs, k=item_vecs,
            v=concate_item_vecs,
            mask=None)

        # [nodes] x [out_dim]
        user_output = dot(concate_user_vecs, self.user_weights, sparse=False)
        item_output = dot(concate_item_vecs, self.item_weights, sparse=False)

        # bias
        if self.bias:
            user_output += self.user_bias
            item_output += self.item_bias

        user_output = self.act(user_output)
        item_output = self.act(item_output)

        #  Combination
        if self.concat:
            user_output = dot(user_output, self.concate_user_weights,
                              sparse=False)
            item_output = dot(item_output, self.concate_item_weights,
                              sparse=False)

            user_output = tf.concat([user_vecs, user_output], axis=1)
            item_output = tf.concat([item_vecs, item_output], axis=1)

        return user_output, item_output


class GASConcatenation(layers.Layer):
    """GCN-based Anti-Spam(GAS) layer for concatenation of comment embedding
    learned by GCN from the Comment Graph and other embeddings learned in
    previous operations.
    """

    def __init__(self, **kwargs):
        super(GASConcatenation, self).__init__(**kwargs)

    def __call__(self, inputs):
        adj_list, concat_vecs = inputs
        # neighbor sample
        ri = tf.nn.embedding_lookup(concat_vecs[2],
                                    tf.cast(adj_list[5], dtype=tf.int32))
        # ri = tf.transpose(tf.random.shuffle(tf.transpose(ri)))
        # ir = tf.slice(ir, [0, 0], [-1, num_samples])

        ru = tf.nn.embedding_lookup(concat_vecs[1],
                                    tf.cast(adj_list[4], dtype=tf.int32))
        # ru = tf.transpose(tf.random.shuffle(tf.transpose(ru)))
        # ru = tf.slice(ru, [0, 0], [-1, num_samples])
        concate_vecs = tf.concat([ri, concat_vecs[0], ru, concat_vecs[3]],
                                 axis=1)
        return concate_vecs
