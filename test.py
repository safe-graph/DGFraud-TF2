"""
This code is attributed to Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from typing import Tuple
import collections

import tensorflow as tf

from algorithms.FdGars.FdGars_main import FdGars_main
from algorithms.GraphConsis.GraphConsis_main import GraphConsis_main
from algorithms.Player2Vec.Player2Vec_main import Player2Vec_main
from algorithms.GraphSage.GraphSage_main import GraphSage_main
from algorithms.GEM.GEM_main import GEM_main
from utils.utils import preprocess_adj, preprocess_feature, sample_mask

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


def load_example_data(meta: bool = False, data: str = 'dblp') -> \
        Tuple[list, np.array, list, np.array]:
    """
    Loading the a small handcrafted data for testing

    :param meta: if True: it loads a HIN with two meta-graphs,
                 if False: it loads a homogeneous graph
    :param data: the example data type, 'dblp' or 'yelp'
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

    if data == 'dblp':
        y = np.array([[0, 1], [1, 0], [1, 0], [1, 0],
                      [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float)
    else:
        y = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.int)

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

    return rownetworks, features, split_ids, y


print("Testing FdGars...")
# load the data
adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
    load_example_data(meta=True, data='dblp')

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

print("Testing Player2Vec...")
# load the data
args.nodes = features.shape[0]

supports = []
for i in range(len(adj_list)):
    hidden = preprocess_adj(adj_list[i])
    supports.append(hidden)
args.class_size = y.shape[1]

# get sparse tensors
for i in range(len(supports)):
    supports[i] = [
        tf.cast(tf.SparseTensor(*supports[i]), dtype=tf.float32)]

Player2Vec_main(supports, features, label, [train_mask, val_mask, test_mask],
                args)

print("Testing GEM...")
adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
    load_example_data(meta=True, data='dblp')

# convert to dense tensors
label = tf.convert_to_tensor(y, dtype=tf.float32)

# initialize the model parameters
args.input_dim = features.shape[1]
args.nodes_num = features.shape[0]
args.class_size = y.shape[1]
args.train_size = len(idx_train)
args.device_num = len(adj_list)
args.hop = 2

features = preprocess_feature(features)
supports = [preprocess_adj(adj) for adj in adj_list]

# get sparse tensors
features = tf.cast(tf.SparseTensor(*features), dtype=tf.float32)
supports = [tf.cast(tf.SparseTensor(*support), dtype=tf.float32) for
            support in supports]

masks = [idx_train, idx_val, idx_test]

GEM_main(supports, features, label, masks, args)

print("Testing GAS...")
exec(open("algorithms/GAS/GAS_main.py").read())

print("Testing SemiGNN...")
exec(open("algorithms/SemiGNN/SemiGNN_main.py").read())

print("Testing GraphSAGE...")
# load the data
adj_list, features, split_ids, y = load_example_data(meta=True, data='yelp')
idx_train, _, idx_val, _, idx_test, _ = split_ids

num_classes = len(set(y))
label = np.array([y]).T
args.nhid = 128
args.sample_sizes = [5, 5]

features = preprocess_feature(features, to_tuple=False)
features = np.array(features.todense())

neigh_dict = collections.defaultdict(list)
for i in range(len(y)):
    neigh_dict[i] = []

# merge all relations into single graph
for net in adj_list:
    nodes1 = net.nonzero()[0]
    nodes2 = net.nonzero()[1]
    for node1, node2 in zip(nodes1, nodes2):
        neigh_dict[node1].append(node2)

neigh_dict = {k: np.array(v, dtype=np.int64)
              for k, v in neigh_dict.items()}

GraphSage_main(neigh_dict, features, label, [idx_train, idx_val, idx_test],
               num_classes, args)

print("Testing GraphConsis...")
# load the data
adj_list, features, split_ids, y = load_example_data(meta=True, data='yelp')
idx_train, _, idx_val, _, idx_test, _ = split_ids

num_classes = len(set(y))
label = np.array([y]).T
args.identity_dim = 0
args.eps = 0.001

features = preprocess_feature(features, to_tuple=False)
features = np.array(features.todense())

# Equation 2 in the paper
features = np.concatenate((features,
                           np.random.rand(features.shape[0],
                                          args.identity_dim)), axis=1)

neigh_dicts = []
for net in adj_list:
    neigh_dict = {}
    for i in range(len(y)):
        neigh_dict[i] = []
    nodes1 = net.nonzero()[0]
    nodes2 = net.nonzero()[1]
    for node1, node2 in zip(nodes1, nodes2):
        neigh_dict[node1].append(node2)
    neigh_dicts.append({k: np.array(v, dtype=np.int64)
                        for k, v in neigh_dict.items()})

GraphConsis_main(neigh_dicts, features, label, [idx_train, idx_val, idx_test],
                 num_classes, args)
