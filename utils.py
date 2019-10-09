import numpy as np
import scipy.sparse as sp
import torch
import pickle
import pandas as pd
import os
import math


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="simul_data"):
    """load dataset from path
    adj is a matrix, features is a dict with times as key, and [n_nodes, 3] for each value"""
    adj = pd.read_csv(os.path.join(path, 'adj.csv'))
    adj = np.array(adj.values)
    with open(os.path.join(path, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    return adj, features


def train_test_split(features, ratio=(0.5, 0.3, )):
    """split the loaded features data into train dataset and test dataset
    train is """
    assert sum(ratio) <= 1
    times = list(features.keys())
    features_train = [features[i] for i in times[0: int(len(times)*ratio[0])]]
    features_val = [features[i] for i in times[int(len(times)*ratio[0]):int(len(times)*(ratio[0]+ratio[1]))]]
    features_test = [features[i] for i in times[int(len(times)*(ratio[0]+ratio[1])):]]
    return features_train, features_val, features_test


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def rmse(y_pre, y_true):
    inputs = y_pre - y_true
    return np.linalg.norm(inputs, ord=2)/math.sqrt(len(inputs))


def mae(y_pre, y_true):
    inputs = y_pre - y_true
    return np.linalg.norm(inputs, ord=1)/len(inputs)


def mare(y_pre, y_true):
    inputs = [(i-j)/j for i, j in zip(y_pre, y_true)]
    return np.linalg.norm(inputs, ord=1)/len(inputs)


def get_losses(target, prediction, method='rmse'):
    if method == 'rmse':
        return rmse(prediction, target)
    elif method == 'mae':
        return mae(prediction, target)
    elif method == 'mare':
        return mare(prediction, target)


def visulization(target, prediction):
    import matplotlib.pyplot as plt
    index = torch.nonzero(target)
    plt.plot(target[index], 'r')
    plt.plot(prediction[index], 'b')
    plt.show()