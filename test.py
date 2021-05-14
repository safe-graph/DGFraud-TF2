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
from utils.utils import preprocess_adj, preprocess_feature, sample_mask

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
