"""
This code is attributed to Yingtong Dou (@YingtongDou), Kay Liu (@kayzliu),
and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import os
import sys
from typing import Tuple
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from utils.utils import pad_adjlist

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))


def load_data_dblp(path: str =
                   'dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat',
                   train_size: int = 0.8, meta: bool = True) -> \
        Tuple[list, np.array, list, np.array]:
    """
    The data loader to load the DBLP heterogeneous information network data
    source: https://github.com/Jhy1993/HAN

    :param path: the local path of the dataset file
    :param train_size: the percentage of training data
    :param meta: if True: it loads a HIN with three meta-graphs,
                 if False: it loads a homogeneous APA meta-graph
    """
    data = sio.loadmat(path)
    truelabels, features = data['label'], data['features'].astype(float)
    N = features.shape[0]

    if not meta:
        rownetworks = [data['net_APA'] - np.eye(N)]
    else:
        rownetworks = [data['net_APA'] - np.eye(N),
                       data['net_APCPA'] - np.eye(N),
                       data['net_APTPA'] - np.eye(N)]

    y = truelabels
    index = np.arange(len(y))
    X_train, X_test, y_train, y_test = \
        train_test_split(index, y, stratify=y, test_size=1 - train_size,
                         random_state=48, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      stratify=y_train,
                                                      test_size=0.2,
                                                      random_state=48,
                                                      shuffle=True)

    split_ids = [X_train, y_train, X_val, y_val, X_test, y_test]

    return rownetworks, features, split_ids, np.array(y)


def load_data_yelp(path: str = 'dataset/YelpChi.mat',
                   train_size: int = 0.8, meta: bool = True) -> \
        Tuple[list, np.array, list, np.array]:
    """
    The data loader to load the Yelp heterogeneous information network data
    source: http://odds.cs.stonybrook.edu/yelpchi-dataset

    :param path: the local path of the dataset file
    :param train_size: the percentage of training data
    :param meta: if True: it loads a HIN with three meta-graphs,
                 if False: it loads a homogeneous rur meta-graph
    """
    data = sio.loadmat(path)
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = truelabels.tolist()[0]

    if not meta:
        rownetworks = [data['net_rur']]
    else:
        rownetworks = [data['net_rur'], data['net_rsr'], data['net_rtr']]

    y = truelabels
    index = np.arange(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index,
                                                        y,
                                                        stratify=y,
                                                        test_size=1-train_size,
                                                        random_state=48,
                                                        shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      stratify=y_train,
                                                      test_size=0.2,
                                                      random_state=48,
                                                      shuffle=True)

    split_ids = [X_train, y_train, X_val, y_val, X_test, y_test]

    return rownetworks, features, split_ids, np.array(y)


def load_example_semi():
    """
    The data loader to load the example data for SemiGNN

    """
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


def load_data_gas():
    """
    The data loader to load the example data for GAS

    """
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
        [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])
    index = range(len(y))

    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y,
                                                        test_size=0.4,
                                                        random_state=48,
                                                        shuffle=True)
    split_ids = [X_train, X_test]
    return adjs, features, split_ids, y
