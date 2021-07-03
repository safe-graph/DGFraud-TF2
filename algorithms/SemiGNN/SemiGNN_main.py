"""
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou),
Zhongzheng Lu(@lzz-hub-dev) and UIC BDSC Lab
DGFraud-TF2 (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers

from algorithms.SemiGNN.SemiGNN import SemiGNN
from utils.data_loader import load_example_semi
from utils.utils import matrix_to_adjlist, random_walks, pairs_to_matrix, \
    get_negative_sampling

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset_str', type=str, default='example',
                    help="['dblp','example']")
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train.')
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--lr', default=0.001, help='learning rate')

# SemiGNN
parser.add_argument('--init_emb_size', default=4,
                    help='initial node embedding size')
parser.add_argument('--semi_encoding1', default=3,
                    help='the first view attention layer unit number')
parser.add_argument('--semi_encoding2', default=2,
                    help='the second view attention layer unit number')
parser.add_argument('--semi_encoding3', default=4,
                    help='one-layer perceptron units')
parser.add_argument('--Ul', default=8, help='labeled users number')
parser.add_argument('--alpha', default=0.5, help='loss alpha')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def SemiGNN_main(adj_list: list,
                 label: tf.Tensor,
                 masks: list,
                 args: argparse.ArgumentParser().parse_args()) -> None:
    """
    Main function to train and test the model

    :param adj_list: a list of the sparse adjacency matrices
    :param label: the label tensor for all nodes
    :param masks: a list of mask tensors to obtain the train, val, test data
    :param args: model arguments
    """
    model = SemiGNN(args.nodes, args.class_size, args.semi_encoding1,
                    args.semi_encoding2, args.semi_encoding3,
                    args.init_emb_size, args.view_num, args.alpha)

    optimizer = optimizers.Adam(lr=args.lr)
    adj_nodelists = [matrix_to_adjlist(adj, pad=False) for adj in
                     adj_list]
    pairs = [random_walks(adj_nodelists[i], 2, 3)
             for i in range(args.view_num)]
    adj_data = [pairs_to_matrix(p, args.nodes) for p in pairs]
    u_i = []
    u_j = []
    graph_label = []
    for adj_nodelist, p in zip(adj_nodelists, pairs):
        u_i_t, u_j_t, graph_label_t = get_negative_sampling(p, adj_nodelist)
        u_i.append(u_i_t)
        u_j.append(u_j_t)
        graph_label.append(graph_label_t)
    u_i = np.concatenate(np.array(u_i))
    u_j = np.concatenate(np.array(u_j))
    graph_label = tf.convert_to_tensor(np.concatenate(graph_label),
                                       dtype=tf.float32)

    # train
    for epoch in range(args.epochs):
        with tf.GradientTape() as tape:
            train_loss, train_acc = model(
                [adj_data, u_i, u_j, graph_label, label, masks[0]])
            print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # test
    test_loss, test_acc = model(
        [adj_data, u_i, u_j, graph_label, label, masks[1]])
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")


if __name__ == "__main__":
    # load the data
    adj_list, features, [X_train, X_test], y = load_example_semi()

    # convert to dense tensors
    label = tf.convert_to_tensor(y, dtype=tf.float32)

    # initialize the model parameters
    args.nodes = features.shape[0]
    args.class_size = y.shape[1]
    args.view_num = len(adj_list)
    masks = [X_train, X_test]

    SemiGNN_main(adj_list, label, [X_train, X_test], args)
