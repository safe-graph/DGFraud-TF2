<p align="center">
    <br>
    <a href="https://image.flaticon.com/icons/svg/1671/1671517.svg">
        <img src="https://github.com/safe-graph/DGFraud-TF2/blob/main/logo.png" width="550"/>
    </a>
    <br>
<p>
<p align="center">
    <a href="https://travis-ci.com/github/safe-graph/DGFraud-TF2">
        <img alt="travis-ci" src="https://travis-ci.com/safe-graph/DGFraud-TF2.svg?token=wicswr4X2g4v8jddTpUv&branch=main">
    </a>
    <a href="https://www.tensorflow.org/install">
        <img alt="Tensorflow" src="https://img.shields.io/badge/tensorflow-2.X-orange">
    </a>
    <a href="https://www.python.org/">
        <img alt="Python" src="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue">
    </a>
    <a href="https://github.com/safe-graph/DGFraud-TF2/archive/main.zip">
        <img alt="PRs" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
    </a>
    <a href="https://github.com/safe-graph/DGFraud-TF2/pulls">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/safe-graph/DGFraud-TF2?include_prereleases">
    </a>
</p>

<h3 align="center">
<p>A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X
</h3>

[Introduction](#introduction) | [Useful Resources](#useful-resources) | [Installation](#installation) |  [Datasets](#datasets) | [User Guide](#user-guide) | [Implemented Models](#implemented-models) | [How to Contribute](#how-to-contribute)


## Introduction

**DGFraud-TF2** is a Graph Neural Network (GNN) based toolbox for fraud detection. It is the Tensorflow 2.X version of [DGFraud](https://github.com/safe-graph/DGFraud), which is implemented using TF 1.X. It integrates the implementation & comparison of state-of-the-art GNN-based fraud detection models. The introduction of implemented models can be found [here](#implemented-models).

We welcome contributions to this repo like adding new fraud detectors and extending the features of the toolbox.

If you use the toolbox in your project, please cite the paper below and the [algorithms](#implemented-models) you used:

CIKM'20 ([PDF](https://arxiv.org/pdf/2008.08692.pdf))
```bibtex
@inproceedings{dou2020enhancing,
  title={Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters},
  author={Dou, Yingtong and Liu, Zhiwei and Sun, Li and Deng, Yutong and Peng, Hao and Yu, Philip S},
  booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM'20)},
  year={2020}
}
```


## Useful Resources
- [UGFraud: An Unsupervised Graph-based Toolbox for Fraud Detection](https://github.com/safe-graph/UGFraud)
- [Graph-based Fraud Detection Paper List](https://github.com/safe-graph/graph-fraud-detection-papers) 
- [Awesome Fraud Detection Papers](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers)
- [PyOD: A Python Toolbox for Scalable Outlier Detection (Anomaly Detection)](https://github.com/yzhao062/pyod)
- [PyODD: An End-to-end Outlier Detection System](https://github.com/datamllab/pyodds)
- [DGL: Deep Graph Library](https://github.com/dmlc/dgl)
- [Realtime Fraud Detection with GNN on DGL](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl)
- [Outlier Detection DataSets (ODDS)](http://odds.cs.stonybrook.edu/)

## Installation
```bash
git clone https://github.com/safe-graph/DGFraud-TF2.git
cd DGFraud-TF2
python setup.py install
```
### Requirements
```bash
* python>=3.6
* tensorflow>=2.0
* numpy>=1.16.4
* scipy>=1.2.0
```
## Datasets

### DBLP
We uses the pre-processed DBLP dataset from [Jhy1993/HAN](https://github.com/Jhy1993/HAN)
You can run the FdGars, Player2Vec, GeniePath and GEM based on the DBLP dataset.
Unzip the archive before using the dataset:
```bash
cd dataset
unzip DBLP4057_GAT_with_idx_tra200_val_800.zip
```

### Example dataset
We implement example graphs for SemiGNN, GAS and GEM in `data_loader.py`. Because those models require unique graph structures or node types, which cannot be found in opensource datasets.


### Yelp dataset
For [GraphConsis](https://arxiv.org/abs/2005.00625) and [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf), we preprocessed [Yelp Spam Review Dataset](http://odds.cs.stonybrook.edu/yelpchi-dataset/) with reviews as nodes and three relations as edges.

The dataset with `.mat` format is located at `/dataset/YelpChi.zip`. The `.mat` file includes:
- `net_rur, net_rtr, net_rsr`: three sparse matrices representing three homo-graphs defined in [GraphConsis](https://arxiv.org/abs/2005.00625) paper;
- `features`: a sparse matrix of 32-dimension handcrafted features;
- `label`: a numpy array with the ground truth of nodes. `1` represents spam and `0` represents benign.

The YelpChi data preprocessing details can be found in our [CIKM'20](https://arxiv.org/pdf/2008.08692.pdf) paper.
To get the complete metadata of the Yelp dataset, please email to [ytongdou@gmail.com](mailto:ytongdou@gmail.com) for inquiry.

## User Guide

### Running the example code
You can find the implemented models in `algorithms` directory. For example, you can run Player2Vec using:
```bash
python Player2Vec_main.py 
```
You can specify parameters for models when running the code.

### Running on your datasets
Have a look at the load_data_dblp() function in utils/utils.py for an example.

In order to use your own data, you have to provide:
* adjacency matrices or adjlists (for GAS);
* a feature matrix
* a label matrix
then split feature matrix and label matrix into testing data and training data.

You can specify a dataset as follows:
```bash
python xx_main.py --dataset your_dataset 
```
or by editing xx_main.py

### The structure of code
The repository is organized as follows:
- `algorithms/` contains the implemented models and the corresponding example code;
- `layers/` contains all GNN layers used by implemented models;
- `dataset/` contains the necessary dataset files;
- `utils/` contains:
    * loading and splitting the data (`data_loader.py`);
    * contains various utilities (`utils.py`).


## Implemented Models

### Model Source

| Model  | Paper  | Venue  | Reference  |
|-------|--------|--------|--------|
| **SemiGNN** | [A Semi-supervised Graph Attentive Network for Financial Fraud Detection](https://arxiv.org/pdf/2003.01171)  | ICDM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/semignn.txt) |
| **Player2Vec** | [Key Player Identification in Underground Forums over Attributed Heterogeneous Information Network Embedding Framework](http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf)  | CIKM 2019  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/player2vec.txt)|
| **GAS** | [Spam Review Detection with Graph Convolutional Networks](https://arxiv.org/abs/1908.10679)  | CIKM 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gas.txt) |
| **FdGars** | [FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System](https://dl.acm.org/citation.cfm?id=3316586)  | WWW 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/fdgars.txt) |
| **GeniePath** | [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://arxiv.org/abs/1802.00910)  | AAAI 2019 | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/geniepath.txt)  |
| **GEM** | [Heterogeneous Graph Neural Networks for Malicious Account Detection](https://arxiv.org/pdf/2002.12307.pdf)  | CIKM 2018 |[BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/gem.txt) |
| **GraphSAGE** | [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)  | NIPS 2017  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/graphsage.txt) |
| **GraphConsis** | [Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection](https://arxiv.org/pdf/2005.00625.pdf)  | SIGIR 2020  | [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/graphconsis.txt) |
| **HACUD** | [Cash-Out User Detection Based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism](https://aaai.org/ojs/index.php/AAAI/article/view/3884)  | AAAI 2019 |  [BibTex](https://github.com/safe-graph/DGFraud/blob/master/reference/hacud.txt) |


### Model Comparison
| Model  | Application  | Graph Type  | Base Model  |
|-------|--------|--------|--------|
| **SemiGNN** | Financial Fraud  | Heterogeneous   | GAT, LINE, DeepWalk |
| **Player2Vec** | Cyber Criminal  | Heterogeneous | GAT, GCN|
| **GAS** | Opinion Fraud  | Heterogeneous | GCN, GAT |
| **FdGars** |  Opinion Fraud | Homogeneous | GCN |
| **GeniePath** | Financial Fraud | Homogeneous | GAT  |
| **GEM** | Financial Fraud  | Heterogeneous |GCN |
| **GraphSAGE** | Opinion Fraud  | Homogeneous   | GraphSAGE |
| **GraphConsis** | Opinion Fraud  | Heterogeneous   | GraphSAGE |
| **HACUD** | Financial Fraud | Heterogeneous | GAT |



## How to Contribute
You are welcomed to contribute to this open-source toolbox. Currently, you can create PR or email to [bdscsafegraph@gmail.com](mailto:bdscsafegraph@gmail.com) for inquiry.
