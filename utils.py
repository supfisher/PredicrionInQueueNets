import numpy as np
import scipy.sparse as sp
import torch
import pickle
import pandas as pd
import os
import math


def load_path(path="simul_data"):
    """load dataset from path
    adj is a matrix, features is a dict with times as key, and [n_nodes, 3] for each value"""
    adj = pd.read_csv(os.path.join(path, 'adj.csv'))
    adj = np.array(adj.values)
    with open(os.path.join(path, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(path, 'targets.pkl'), 'rb') as f:
        targets = pickle.load(f)
    return adj, features, targets


def train_test_split(features, targets, ratio=(0.5, 0.3, )):
    """split the loaded features data into train dataset and test dataset
    features shape: key: times, values: (num_nodes, node_feature)
    target feature: key: times, values: (target_feature)"""
    Id={'q_id': 0, 'service_time': 1, 'response_time': 2}
    assert sum(ratio) <= 1
    times = sorted(list(features.keys()))
    features_train = [features[i] for i in times[0: int(len(times)*ratio[0])]]
    features_val = [features[i] for i in times[int(len(times)*ratio[0]):int(len(times)*(ratio[0]+ratio[1]))]]
    features_test = [features[i] for i in times[int(len(times)*(ratio[0]+ratio[1])):]]

    targets_train = [targets[i][Id['service_time'],] for i in times[0: int(len(times) * ratio[0])]]
    targets_val = [targets[i][Id['service_time'],] for i in times[int(len(times) * ratio[0]):int(len(times) * (ratio[0] + ratio[1]))]]
    targets_test = [targets[i][Id['service_time'],] for i in times[int(len(times) * (ratio[0] + ratio[1])):]]
    return features_train, features_val, features_test, targets_train, targets_val, targets_test


def nomalization(targets):
    """the shape of targets: (len(times), target_feature)"""
    mu = torch.mean(targets, dim=0)
    std = torch.std(targets, dim=0)
    targets = (targets-mu)/std
    return targets, mu, std

def generate_targets(targets, pre_len, tar_len, args):
    """targets output is a (len(targets)-pre_len-tar_len, tar_len, target_features)"""
    len_feat = len(targets)
    output = torch.empty((len(targets)-pre_len-tar_len, tar_len, targets.shape[-1]))
    for i in range(len_feat-pre_len-tar_len):
        output[i] = targets[i+pre_len-1:i+pre_len+tar_len-1]
    return output


def data_loader(features, targets, batch_size, args):
    """features:(len(times), num_nodes, node_feature)
    targets: (len(features)-pre_len-tar_len, tar_len, target_features)
    retrun: (data, target)
    data_batch:(pre_len, batch_size, num_nodes, node_feature)
    target_batch:(tar_len, batch_size, target_features)"""
    data_batch = torch.empty(args.pre_len, args.batch_size, args.num_nodes, args.node_features)
    target_batch = torch.empty(args.tar_len, args.batch_size, targets.shape[-1])

    orders = torch.randperm(len(features) - args.pre_len - args.tar_len)

    for i in range(0, len(orders), batch_size):
        orders_chunk = orders[i:min(i + batch_size, len(orders) - 1)]
        for j, id in enumerate(orders_chunk):
            data_batch[:,j,:,:] = features[id:id+args.pre_len,:,:]
            target_batch[:,j,:] = targets[id]
        yield data_batch, target_batch


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
    inputs = []
    for i,j in zip(y_pre, y_true):
        if j != 0:
            inputs.append((i-j)/j)
        else:
            inputs.append((i - 0) / 1)
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
    plt.plot(target, 'r')
    plt.plot(prediction, 'b')
    plt.show()