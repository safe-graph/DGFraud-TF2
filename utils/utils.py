"""
This code is attributed to Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

from typing import Tuple
import scipy.sparse as sp
import numpy as np


def sparse_to_tuple(sparse_mx: sp.coo_matrix) -> Tuple[np.array, np.array,
                                                       np.array]:
    """
    Convert sparse matrix to tuple representation.

    :param sparse_mx: the graph adjacency matrix in scipy sparse matrix format
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj: np.array) -> sp.coo_matrix:
    """
    Symmetrically normalize adjacency matrix
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn

    :param adj: the graph adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    Preprocessing of adjacency matrix for simple GCN model
    and conversion to tuple representation.
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn

    :param adj: the graph adjacency matrix
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_feature(features: np.array, to_tuple: bool = True) -> \
        Tuple[np.array, np.array, np.array]:
    """
    Row-normalize feature matrix and convert to tuple representation
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn

    :param features: the node feature matrix
    :param to_tuple: whether cast the feature matrix to scipy sparse tuple
    """
    features = sp.lil_matrix(features)
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    if to_tuple:
        return sparse_to_tuple(features)
    else:
        return features


def sample_mask(idx: np.array, n_class: int) -> np.array:
    """
    Create mask for GCN.
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn

    :param idx: the train/val/test indices
    :param n_class: the number of classes for the data
    """
    mask = np.zeros(n_class)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
