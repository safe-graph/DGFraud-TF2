"""
This code is attributed to Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import os
import sys
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from typing import Tuple

import tensorflow as tf

from algorithms.FdGars.FdGars_main import FdGars_main
from algorithms.GAS.GAS_main import GAS_main
from algorithms.Player2Vec.Player2Vec_main import Player2Vec_main
from algorithms.SemiGNN.SemiGNN_main import SemiGNN_main
from utils.utils import preprocess_adj, preprocess_feature, sample_mask, \
    pad_adjlist

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='weight decay')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units in GCN')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def load_example_data(meta: bool = False) -> \
        Tuple[list, np.array, list, np.array]:
    """
    Loading the a small handcrafted data for testing

    :param meta: if True: it loads a HIN with two meta-graphs,
                 if False: it loads a homogeneous graph
    """

    features = np.array([[1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 1, 1, 0],
                         [0, 1, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1, 0, 1],
                         [1, 0, 1, 1, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 1, 1]
                         ], dtype=np.float)
    if meta:
        # a heterogeneous information network
        # with two meta-paths
        rownetworks = [np.array([[1, 0, 0, 1, 0, 1, 1, 1],
                                 [1, 0, 0, 1, 1, 1, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 0, 1, 0, 0],
                                 [1, 0, 0, 1, 1, 1, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 1, 1, 1, 0]], dtype=np.float),
                       np.array([[1, 0, 0, 0, 0, 1, 1, 1],
                                 [0, 1, 0, 0, 1, 1, 0, 0],
                                 [0, 1, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 1, 1, 0, 0, 1],
                                 [1, 1, 0, 1, 1, 0, 0, 0],
                                 [1, 0, 0, 1, 0, 1, 1, 1],
                                 [1, 0, 0, 1, 1, 1, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float)]
    else:
        # a homogeneous graph
        rownetworks = [np.array([[1, 0, 0, 1, 0, 1, 1, 1],
                                 [1, 0, 0, 1, 1, 1, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 0, 1, 0, 0],
                                 [1, 0, 0, 1, 1, 1, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 1, 1, 1, 0]], dtype=np.float)]

    y = np.array([[0, 1], [1, 0], [1, 0], [1, 0],
                  [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float)

    index = range(len(y))

    X_train, X_test, y_train, y_test = train_test_split(index, y,
                                                        test_size=0.375,
                                                        random_state=48,
                                                        shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2,
                                                      random_state=48,
                                                      shuffle=True)

    split_ids = [X_train, y_train, X_val, y_val, X_test, y_test]

    return rownetworks, features, split_ids, np.array(y)


def load_data_gas():
    # example data for GAS
    # construct U-E-I network
    user_review_adj = [[0, 1], [2], [3], [5], [4, 6]]
    user_review_adj = pad_adjlist(user_review_adj)
    user_item_adj = [[0, 1], [0], [0], [2], [1, 2]]
    user_item_adj = pad_adjlist(user_item_adj)
    item_review_adj = [[0, 2, 3], [1, 4], [5, 6]]
    item_review_adj = pad_adjlist(item_review_adj)
    item_user_adj = [[0, 1, 2], [0, 4], [3, 4]]
    item_user_adj = pad_adjlist(item_user_adj)
    review_item_adj = [0, 1, 0, 0, 1, 2, 2]
    review_user_adj = [0, 0, 1, 2, 4, 3, 4]

    # initialize review_vecs
    review_vecs = np.array([[1, 0, 0, 1, 0],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1]])

    # initialize user_vecs and item_vecs with user_review_adj and
    # item_review_adj
    # for example, u0 has r1 and r0, then we get the first line of user_vecs:
    # [1, 1, 0, 0, 0, 0, 0]
    user_vecs = np.array([[1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0, 1]])
    item_vecs = np.array([[1, 0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 1]])
    features = [review_vecs, user_vecs, item_vecs]

    # initialize the Comment Graph
    homo_adj = [[1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 0],
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0]]

    adjs = [user_review_adj, user_item_adj, item_review_adj, item_user_adj,
            review_user_adj, review_item_adj, homo_adj]

    y = np.array(
        [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])  # get
    index = range(len(y))

    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y,
                                                        test_size=0.4,
                                                        random_state=48,
                                                        shuffle=True)
    split_ids = [X_train, X_test]
    return adjs, features, split_ids, y


def load_example_semi():
    # example data for SemiGNN
    features = np.array([[1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1, 0, 1],
                         [1, 0, 1, 1, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1]])
    # Here we use binary matrix as adjacency matrix,
    # weighted matrix is acceptable as well
    rownetworks = [np.array([[1, 0, 0, 1, 0, 1, 1, 1],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0, 1, 0, 0],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 1, 1, 1, 0]]),
                   np.array([[1, 0, 0, 0, 0, 1, 1, 1],
                             [0, 1, 0, 0, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0, 1],
                             [1, 1, 0, 1, 1, 0, 0, 0],
                             [1, 0, 0, 1, 0, 1, 1, 1],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1]])]
    y = np.array(
        [[0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1]])
    index = range(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y,
                                                        test_size=0.2,
                                                        random_state=48,
                                                        shuffle=True)
    # test_size=0.25  batch——size=2
    split_ids = [X_train, X_test]

    return rownetworks, features, split_ids, y


# testing FdGars
# load the data
adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
    load_example_data(meta=True)

# convert to dense tensors
train_mask = tf.convert_to_tensor(sample_mask(idx_train, y.shape[0]))
val_mask = tf.convert_to_tensor(sample_mask(idx_val, y.shape[0]))
test_mask = tf.convert_to_tensor(sample_mask(idx_test, y.shape[0]))
label = tf.convert_to_tensor(y)

# get sparse tuples
features = preprocess_feature(features)
support = preprocess_adj(adj_list[0])

# initialize the model parameters
args.input_dim = features[2][1]
args.output_dim = y.shape[1]
args.train_size = len(idx_train)
args.num_features_nonzero = features[1].shape

# get sparse tensors
features = tf.SparseTensor(*features)
support = [tf.cast(tf.SparseTensor(*support), dtype=tf.float32)]

FdGars_main(support, features, label, [train_mask, val_mask, test_mask], args)

# testing GAS
# load the data
adj_list, features, [X_train, X_test], y = load_data_gas()

# pre_process and convert to dense tensors
r_support, r_feature = adj_list, features
r_feature = np.array(features[0], dtype=float)
r_support = np.array(adj_list[6], dtype=float)
features[0] = tf.convert_to_tensor(features[0], dtype=tf.float32)
features[1] = tf.convert_to_tensor(features[1], dtype=tf.float32)
features[2] = tf.convert_to_tensor(features[2], dtype=tf.float32)
label = tf.convert_to_tensor(y, dtype=tf.float32)

# get sparse tuples
r_feature = preprocess_feature(r_feature)
r_support = preprocess_adj(r_support)

# initialize the model parameters
args.reviews_num = features[0].shape[0]
args.class_size = y.shape[1]
args.input_dim_i = features[2].shape[1]
args.input_dim_u = features[1].shape[1]
args.input_dim_r = features[0].shape[1]
args.input_dim_r_gcn = r_feature[2][1]
args.num_features_nonzero = r_feature[1].shape
args.h_u_size = adj_list[0].shape[1] * (
        args.input_dim_r + args.input_dim_u)
args.h_i_size = adj_list[2].shape[1] * (
        args.input_dim_r + args.input_dim_i)
args.output_dim1 = 64
args.output_dim2 = 64
args.output_dim3 = 64
args.output_dim4 = 64
args.output_dim5 = 64
args.gcn_dim = 5

# get sparse tensors
r_feature = tf.SparseTensor(*r_feature)
r_support = [tf.cast(tf.SparseTensor(*r_support), dtype=tf.float32)]
masks = [X_train, X_test]

GAS_main(adj_list, r_support, features, r_feature, label, [X_train, X_test],
         args)

# testing Player2Vec
# load the data
adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
    load_example_data(meta=True)
args.nodes = features.shape[0]
# convert to dense tensors
train_mask = tf.convert_to_tensor(sample_mask(idx_train, y.shape[0]))
val_mask = tf.convert_to_tensor(sample_mask(idx_val, y.shape[0]))
test_mask = tf.convert_to_tensor(sample_mask(idx_test, y.shape[0]))
label = tf.convert_to_tensor(y)

# get sparse tuples
features = preprocess_feature(features)
supports = []
for i in range(len(adj_list)):
    hidden = preprocess_adj(adj_list[i])
    supports.append(hidden)

# initialize the model parameters
args.input_dim = features[2][1]
args.output_dim = y.shape[1]
args.train_size = len(idx_train)
args.class_size = y.shape[1]
args.num_features_nonzero = features[1].shape

# get sparse tensors
features = tf.SparseTensor(*features)
for i in range(len(supports)):
    supports[i] = [
        tf.cast(tf.SparseTensor(*supports[i]), dtype=tf.float32)]

Player2Vec_main(supports, features, label, [train_mask, val_mask, test_mask],
                args)

# testing SemiGNN
# load the data
adj_list, features, [X_train, X_test], y = load_example_semi()

# convert to dense tensors
label = tf.convert_to_tensor(y, dtype=tf.float32)

# initialize the model parameters
args.nodes = features.shape[0]
args.class_size = y.shape[1]
args.meta = len(adj_list)
args.semi_encoding1 = 3
args.semi_encoding2 = 2
args.semi_encoding3 = 4
args.init_emb_size = 4
args.alpha = 0.5
args.lamtha = 0.5

# get masks
masks = [X_train, X_test]

SemiGNN_main(adj_list, label, [X_train, X_test], args)
