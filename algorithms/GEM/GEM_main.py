"""
This code is due to Zhiqin Yang (@visitworld123) Yutong Deng (@yutongD),
Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import argparse
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import optimizers

from algorithms.GEM.GEM import GEM
from utils.data_loader import load_data_dblp
from utils.utils import preprocess_feature, preprocess_adj

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset_str', type=str, default='dblp',
                    help="['dblp','example']")
parser.add_argument('--train_size', type=float, default=0.2,
                    help='training set percentage')
parser.add_argument('--epochs', type=int, default=44,
                    help='Number of epochs to train.')
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--lr', default=0.002, help='learning rate')

# GEM
parser.add_argument('--hop', default=2,
                    help='number of hops of neighbors to be aggregated')
parser.add_argument('--output_dim', default=128, help='gem layer unit')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def GEM_main(supports: list, features: tf.SparseTensor,
             label: tf.Tensor, masks: list, args):
    """
    @param supports: a list of the sparse adjacency matrix
    @param features: the feature of the sparse tensor for all nodes
    @param label: the label tensor for all nodes
    @param masks: a list of mask tensors to obtain the train-val-test data
    @param args: additional parameters
    """
    model = GEM(args.input_dim, args.output_dim, args)
    optimizer = optimizers.Adam(lr=args.lr)

    # train
    for _ in tqdm(range(args.epochs)):
        with tf.GradientTape() as tape:
            train_loss, train_acc = model(
                [supports, features, label, masks[0]])

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # validation
        val_loss, val_acc = model([supports, features, label, masks[1]])
        print(
            f"train_loss: {train_loss:.4f},"
            f" train_acc: {train_acc:.4f},"
            f"val_loss: {val_loss:.4f},"
            f"val_acc: {val_acc:.4f}")

    # test
    test_loss, test_acc = model([supports, features, label, masks[2]])
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")


if __name__ == "__main__":
    adj_list, features, [idx_train, _, idx_val, _, idx_test, _], y = \
        load_data_dblp(meta=True, train_size=args.train_size)

    # convert to dense tensors
    label = tf.convert_to_tensor(y, dtype=tf.float32)

    # initialize the model parameters
    args.input_dim = features.shape[1]
    args.nodes_num = features.shape[0]
    args.class_size = y.shape[1]
    args.train_size = len(idx_train)
    args.device_num = len(adj_list)

    features = preprocess_feature(features)
    supports = [preprocess_adj(adj) for adj in adj_list]

    # get sparse tensors
    features = tf.cast(tf.SparseTensor(*features), dtype=tf.float32)
    supports = [tf.cast(tf.SparseTensor(*support), dtype=tf.float32) for
                support in supports]

    masks = [idx_train, idx_val, idx_test]

    GEM_main(supports, features, label, masks, args)
