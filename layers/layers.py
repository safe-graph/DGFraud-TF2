"""
This code is attributed to Yingtong Dou (@YingtongDou),
Zhongzheng Lu(@lzz-hub-dev), Kay Liu (@kayzliu) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

from typing import Callable, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import GlorotUniform


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
            w = self.add_weight('weight' + str(i), [input_dim, output_dim],
                                dtype=tf.float32)
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_weight('bias', [output_dim], dtype=tf.float32)

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

    :param input_dim: the input dimension
    :param attention_size: the number of meta_paths
    :param v_type：activation function type
    :param bias: whether add bias
    """

    def __init__(self, input_dim: int, num_nodes: int, attention_size: int,
                 v_type: str = 'relu', bias: bool = True,
                 **kwargs: Optional) -> None:
        super().__init__(**kwargs)

        self.w_omega = tf.Variable(tf.random.uniform([num_nodes * input_dim,
                                                      attention_size]))
        self.u_omega = tf.Variable(tf.random.uniform([attention_size]))
        self.bias = bias
        self.activation = v_type

        if self.bias:
            self.b_omega = tf.Variable(tf.random.uniform([attention_size]))

    def call(self, inputs: tf.Tensor, return_weights: bool = False,
             joint_type: str = 'weighted_sum', multi_view: bool = True) -> \
            Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Obtain attention value between different meta_path

        :param inputs: the information passed to next layers
        :param return_weights: the output whether return weights
        :param joint_type: the way of calculating output
        :param multi_view: whether it's a multiple view
        """
        if multi_view:
            inputs = tf.expand_dims(inputs, 0)

        v = tf.tensordot(inputs, self.w_omega, axes=1)
        if self.bias:
            v += self.b_omega
        if self.activation == 'tanh':
            v = tf.tanh(v)
        if self.activation == 'relu':
            v = tf.nn.relu(v)
        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')
        alpha = tf.nn.softmax(vu)

        if joint_type == 'weighted_sum':
            output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        if joint_type == 'concatenation':
            output = tf.concat(inputs * tf.expand_dims(alpha, -1), 2)

        if not return_weights:
            return output
        else:
            return output, alpha


class NodeAttention(layers.Layer):
    """ Node level attention for SemiGNN.

    :param input_dim: the input dimension
    :param view_num: the number of views
    """

    def __init__(self, input_dim: int,
                 **kwargs: Optional) -> None:
        super().__init__(**kwargs)

        self.H_v = tf.Variable(tf.random.normal([input_dim, 1], stddev=0.1))

    def call(self, inputs: list, return_weights: bool = False) -> \
            Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Obtain attention value between nodes

        :param inputs: the information passed to next layers
        :param return_weights: the output whether return weights
        """

        # convert adj to sparse tensor
        emb, adj = inputs
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(adj, zero)
        indices = tf.where(where)
        values = tf.gather_nd(adj, indices)
        adj = tf.SparseTensor(indices=indices,
                              values=values,
                              dense_shape=adj.shape)
        v = tf.cast(adj, tf.float32) * tf.squeeze(
            tf.tensordot(emb, self.H_v, axes=1))

        alpha = tf.sparse.softmax(v)  # [nodes,nodes]
        output = tf.sparse.sparse_dense_matmul(alpha, emb)

        if not return_weights:
            return output
        else:
            return output, alpha


class ViewAttention(layers.Layer):
    """ View level attention implementation for SemiGNN

        :param encoding: a list of MLP encoding sizes for each view
        :param layer_size: the number of view attention layer
        :param view_num: the number of views
    """

    def __init__(self, encoding: list, layer_size: int,
                 view_num: int, **kwargs: Optional) -> None:
        super().__init__(**kwargs)

        self.encoding = encoding
        self.view_num = view_num
        self.layer_size = layer_size

        # Eq. (2) in SemiGNN paper
        self.dense_layers = []
        for _ in range(view_num):
            mlp = Sequential()
            for l in range(layer_size):
                mlp.add(tf.keras.layers.Dense(encoding[l], activation="relu"))
            self.dense_layers.append(mlp)

        self.phi = []
        for _ in range(view_num):
            self.phi.append(tf.Variable(tf.random.normal([encoding[-1], 1],
                                                         stddev=0.1)))

    def call(self, inputs: tf.Tensor, return_weights=False) -> \
            Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Obtain attention value between different views.
        Eq. (3) and Eq. (4) in the paper

        :param inputs: the information passed to next layers
        :param return_weights: the output whether return weights
        """

        h = []
        h_phi = []
        for v in range(self.view_num):
            h_v = self.dense_layers[v](inputs[v])
            h.append(h_v)
            h_phi.append(tf.matmul(h_v, self.phi[v]))

        h, h_phi = tf.concat(h, 0), tf.concat(h_phi, 0)

        h = tf.reshape(h, [self.view_num, inputs[0].shape[0],
                           self.encoding[-1]])
        h_phi = tf.reshape(h_phi, [self.view_num, inputs[0].shape[0]])

        alpha = tf.nn.softmax(h_phi, axis=0)
        alpha = tf.expand_dims(alpha, axis=2)
        alpha = tf.repeat(alpha, self.encoding[-1], axis=-1)

        # Eq. (4)
        output = tf.reshape(alpha * h,
                            [inputs[0].shape[0],
                             self.encoding[-1] * self.view_num])

        if not return_weights:
            return output
        else:
            return output, alpha


def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor,
                                 v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Obtain attention value in one embedding

    :param q: original embedding
    :param k: original embedding
    :param v：embedding after aggregate neighbour feature
    :param mask: whether use mask
    """
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention = qk / tf.math.sqrt(dk)
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
        """
        :param input_dim: the dimension of input
        :param output_dim: the dimension of output
        """
        super(ConcatenationAggregator, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.concat = concat
        self.con_agg_weights = self.add_weight('con_agg_weights',
                                               [input_dim, output_dim],
                                               dtype=tf.float32)

    def call(self, inputs):
        """
        :param inputs: the information passed to next layers
        """
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


class SageMeanAggregator(layers.Layer):
    """ GraphSAGE Mean Aggregation Layer
    Parts of this code file were originally forked from
    https://github.com/subbyte/graphsage-tf2
    """

    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight(name=kwargs["name"] + "_weight",
                                 shape=(src_dim * 2, dst_dim),
                                 dtype=tf.float32,
                                 initializer=GlorotUniform,
                                 trainable=True
                                 )

    def call(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat):
        """
        :param tensor dstsrc_features: the embedding from the previous layer
        :param tensor dstsrc2dst: 1d index mapping
                      (prepraed by minibatch generator)
        :param tensor dstsrc2src: 1d index mapping
                      (prepraed by minibatch generator)
        :param tensor dif_mat: 2d diffusion matrix
                      (prepraed by minibatch generator)
        """
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        aggregated_features = tf.matmul(dif_mat, src_features)
        concatenated_features = tf.concat([aggregated_features, dst_features],
                                          1)
        x = tf.matmul(concatenated_features, self.w)
        return self.activ_fn(x)


class ConsisMeanAggregator(SageMeanAggregator):
    """ GraphConsis Mean Aggregation Layer Inherited SageMeanAggregator
    Parts of this code file were originally forked from
    https://github.com/subbyte/graphsage-tf2
    """

    def __init__(self, src_dim, dst_dim, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(src_dim, dst_dim, activ=False, **kwargs)

    def __call__(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat,
                 relation_vec, attention_vec):
        """
        :param tensor dstsrc_features: the embedding from the previous layer
        :param tensor dstsrc2dst: 1d index mapping
                      (prepraed by minibatch generator)
        :param tensor dstsrc2src: 1d index mapping
                      (prepraed by minibatch generator)
        :param tensor dif_mat: 2d diffusion matrix
                      (prepraed by minibatch generator)
        :param tensor relation_vec: 1d corresponding relation vector
        :param tensor attention_vec: 1d layers shared attention weights vector
        """
        # Equation 5,6 in the paper
        x = super().__call__(dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat)
        relation_features = tf.tile([relation_vec], [x.shape[0], 1])
        alpha = tf.matmul(tf.concat([x, relation_features], 1), attention_vec)
        alpha = tf.tile(alpha, [1, x.shape[-1]])
        x = tf.multiply(alpha, x)

        return x


class AttentionAggregator(layers.Layer):
    """This layer equals to equation (5) and equation (8) in
    paper 'Spam Review Detection with Graph Convolutional Networks.'
    """

    def __init__(self, input_dim1, input_dim2, input_dim3, input_dim4,
                 output_dim, dropout=0., bias=False, act=tf.nn.relu,
                 concat=False, **kwargs):
        """
        :param input_dim1: input dimension in user layer
        :param input_dim2: input dimension in item layer
        :param output_dim: output dimension
        :param hid_dim: hidden dimension
        """
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        self.user_neigh_weights = self.add_weight('user_neigh_weights',
                                                  [input_dim1, output_dim],
                                                  dtype=tf.float32)
        self.item_neigh_weights = self.add_weight('item_neigh_weights',
                                                  [input_dim2, output_dim],
                                                  dtype=tf.float32)
        self.center_user_weights = self.add_weight('center_user_weights',
                                                   [input_dim3, output_dim],
                                                   dtype=tf.float32)
        self.center_item_weights = self.add_weight('center_item_weights',
                                                   [input_dim4, output_dim],
                                                   dtype=tf.float32)

        if self.bias:
            self.user_bias = self.add_weight('user_bias', [self.output_dim],
                                             dtype=tf.float32)
            self.item_bias = self.add_weight('item_bias', [self.output_dim],
                                             dtype=tf.float32)

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim

    def call(self, inputs):
        """
        :param inputs: the information passed to next layers
        """
        adj_list, features = inputs

        review_vecs = tf.nn.dropout(features[0], self.dropout)
        user_vecs = tf.nn.dropout(features[1], self.dropout)
        item_vecs = tf.nn.dropout(features[2], self.dropout)

        # num_samples = self.adj_info[4]

        # neighbor sample
        ur = tf.nn.embedding_lookup(review_vecs,
                                    tf.cast(adj_list[0], dtype=tf.int32))
        ur = tf.transpose(tf.random.shuffle(tf.transpose(ur)))

        ri = tf.nn.embedding_lookup(item_vecs,
                                    tf.cast(adj_list[1], dtype=tf.int32))
        ri = tf.transpose(tf.random.shuffle(tf.transpose(ri)))

        ir = tf.nn.embedding_lookup(review_vecs,
                                    tf.cast(adj_list[2], dtype=tf.int32))
        ir = tf.transpose(tf.random.shuffle(tf.transpose(ir)))

        ru = tf.nn.embedding_lookup(user_vecs,
                                    tf.cast(adj_list[3], dtype=tf.int32))
        ru = tf.transpose(tf.random.shuffle(tf.transpose(ru)))

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
        concate_user_vecs, _ = scaled_dot_product_attention(
            q=user_vecs, k=user_vecs,
            v=concate_user_vecs)
        concate_item_vecs, _ = scaled_dot_product_attention(
            q=item_vecs, k=item_vecs,
            v=concate_item_vecs)

        # [nodes] x [out_dim]
        user_output = dot(concate_user_vecs, self.user_neigh_weights, sparse=False)
        item_output = dot(concate_item_vecs, self.item_neigh_weights, sparse=False)

        # bias
        if self.bias:
            user_output += self.user_bias
            item_output += self.item_bias

        user_output = self.act(user_output)
        item_output = self.act(item_output)

        #  Combination Eq. (8) in the paper
        if self.concat:
            user_vecs = dot(user_vecs, self.center_user_weights,
                            sparse=False)
            item_vecs = dot(item_vecs, self.center_item_weights,
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
        """
        :param inputs: the information passed to next layers
        """
        adj_list, concat_vecs = inputs
        # neighbor sample
        ri = tf.nn.embedding_lookup(concat_vecs[2],
                                    tf.cast(adj_list[5], dtype=tf.int32))

        ru = tf.nn.embedding_lookup(concat_vecs[1],
                                    tf.cast(adj_list[4], dtype=tf.int32))

        concate_vecs = tf.concat([ri, concat_vecs[0], ru, concat_vecs[3]],
                                 axis=1)
        return concate_vecs


class GEMLayer(layers.Layer):
    """
    This layer equals to the equation (8) in
    paper 'Heterogeneous Graph Neural Networks
    for Malicious Account Detection.'
    """

    def __init__(self, nodes_num, input_dim, output_dim, device_num, **kwargs):
        super(GEMLayer, self).__init__(**kwargs)

        self.nodes_num = nodes_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device_num = device_num
        self.W = self.add_weight('weight', [input_dim, output_dim],
                                 dtype=tf.float32)
        self.V = self.add_weight('V', [output_dim, output_dim],
                                 dtype=tf.float32)
        self.alpha = self.add_weight('alpha', [self.device_num, 1],
                                     dtype=tf.float32)

    def call(self, inputs):
        """
        :@param inputs: include x, support, h
        x: the node feature
        support: a list of the sparse adjacency Tensor
        h: the hidden layer Tensor
        """
        x, support_, h = inputs
        h1 = dot(x, self.W, sparse=True)
        h2 = []

        for d in range(self.device_num):
            ahv = dot(dot(support_[d], h, sparse=True), self.V, sparse=False)
            h2.append(ahv)

        h2 = tf.concat(h2, 0)
        h2 = tf.reshape(h2, [self.device_num,
                             self.nodes_num * self.output_dim])
        h2 = tf.transpose(h2, [1, 0])
        h2 = tf.reshape(tf.matmul(h2, tf.nn.softmax(self.alpha)),
                        [self.nodes_num, self.output_dim])

        return tf.nn.relu(h1 + h2)
