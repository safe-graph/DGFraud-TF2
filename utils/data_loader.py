"""
This code is attributed to Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""

import os
import sys

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))


def load_data_dblp(path='dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat',
                   train_size=0.8, meta=True):
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
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y,
                                                        test_size=1-train_size,
                                                        random_state=48,
                                                        shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      stratify=y_train,
                                                      test_size=0.2,
                                                      random_state=48,
                                                      shuffle=True)

    split_ids = [X_train, y_train, X_val, y_val, X_test, y_test]

    return rownetworks, features, split_ids, np.array(y)
