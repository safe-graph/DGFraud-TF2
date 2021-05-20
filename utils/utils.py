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


def pad_adjlist(x_data):
    # Get lengths of each row of data
    lens = np.array([len(x_data[i]) for i in range(len(x_data))])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    padded = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        padded[i] = np.random.choice(x_data[i], mask.shape[1])
    padded[mask] = np.hstack((x_data[:]))
    return padded


def matrix_to_adjlist(M, pad=True):
    adjlist = []
    for i in range(len(M)):
        adjline = [i]
        for j in range(len(M[i])):
            if M[i][j] == 1:
                adjline.append(j)
        adjlist.append(adjline)
    if pad:
        adjlist = pad_adjlist(adjlist)
    return adjlist


def pairs_to_matrix(pairs, nodes):
    M = np.zeros((nodes, nodes))
    for i, j in pairs:
        M[i][j] = 1
    return M


# Random walk on graph
def generate_random_walk(adjlist, start, walklength):
    t = 1
    walk_path = np.array([start])
    while t <= walklength:
        neighbors = adjlist[start]
        current = np.random.choice(neighbors)
        walk_path = np.append(walk_path, current)
        start = current
        t += 1
    return walk_path


#  sample multiple times for each node
def random_walks(adjlist, numerate, walklength):
    nodes = range(0, len(adjlist))  # node index starts from zero
    walks = []

    for n in range(numerate):
        for node in nodes:
            walks.append(generate_random_walk(adjlist, node, walklength))
    pairs = []
    for i in range(len(walks)):
        for j in range(1, len(walks[i])):
            pair = [walks[i][0], walks[i][j]]
            pairs.append(pair)
    return pairs


def negative_sampling(adj_nodelist):
    degree = [len(neighbors) for neighbors in adj_nodelist]
    node_negative_distribution = np.power(np.array(degree, dtype=np.float32),
                                          0.75)
    node_negative_distribution /= np.sum(node_negative_distribution)
    node_sampling = AliasSampling(prob=node_negative_distribution)
    return node_negative_distribution, node_sampling


def get_negative_sampling(pairs, adj_nodelist, Q=3, node_sampling='atlas'):
    num_of_nodes = len(adj_nodelist)  # 8
    u_i = []
    u_j = []
    graph_label = []
    node_negative_distribution, nodesampling = negative_sampling(adj_nodelist)
    for index in range(0, num_of_nodes):
        u_i.append(pairs[index][0])
        u_j.append(pairs[index][1])
        graph_label.append(1)
        for i in range(Q):
            while True:
                if node_sampling == 'numpy':
                    negative_node = np.random. \
                        choice(num_of_nodes, node_negative_distribution)
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
                elif node_sampling == 'atlas':
                    negative_node = nodesampling.sampling()
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
                elif node_sampling == 'uniform':
                    negative_node = np.random.randint(0, num_of_nodes)
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
            u_i.append(pairs[index][0])
            u_j.append(negative_node)
            graph_label.append(-1)
    graph_label = np.array(graph_label, dtype=np.int32)
    graph_label = graph_label.reshape(graph_label.shape[0], 1)
    return u_i, u_j, graph_label


# Reference: https://en.wikipedia.org/wiki/Alias_method
class AliasSampling:

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
