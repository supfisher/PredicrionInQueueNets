import numpy as np
import random
from sklearn import metrics
import torch
import pickle
import pandas as pd
import os
import math
from torch_geometric.data import Data, Batch
from enum import Enum
import shutil

class Index(Enum):
    arrival = 0
    service = 1
    departure = 2
    num_queued = 3
    num_total = 4
    q_id = 5


def adjMat2edgeIndex(adj):
    edgeIndex = torch.tensor(np.where(adj != 0)).long().contiguous()
    assert edgeIndex.shape[0] == 2
    eyes = torch.eye(adj.shape[0], adj.shape[1])
    adj += eyes
    D = torch.sum(adj, dim=1).diagflat().sqrt()
    adj = torch.mm(D, torch.mm(adj, D))
    return edgeIndex, adj


def normalizationMat(A):
    """given the original mat A, and return a normalized mat defined by:
    A = \Bar{A}- D^{-1/2}\tildle{A}D^{-1/2}, where \tildle{A} = A+I_n"""
    I = torch.eye(A.shape[0])
    A += I
    D = torch.sum(A, dim=0)
    D = torch.sqrt(D)
    D = torch.diagflat(D)
    return torch.mm(torch.mm(D, A), D)


def load_dataset(path="../simul_data"):

    features_train = torch.load(os.path.join(path, 'features_train.pkl'))
    features_val = torch.load(os.path.join(path, 'features_val.pkl'))
    features_test = torch.load(os.path.join(path, 'features_test.pkl'))
    targets_train = torch.load(os.path.join(path, 'targets_train.pkl'))
    targets_val = torch.load(os.path.join(path, 'targets_val.pkl'))
    targets_test = torch.load(os.path.join(path, 'targets_test.pkl'))
    adj = pd.read_csv(os.path.join(path, 'adj.csv'))
    adj = torch.tensor(adj.values).float()
    adj[adj > 0] = 1
    edge_index, adj = adjMat2edgeIndex(adj)

    return adj, edge_index, features_train, features_val, features_test, targets_train, targets_val, targets_test


def normalization(data, mu=None, std=None):
    """the first dimision of data is len(times)"""
    data = torch.tensor(data)
    if mu is None and std is None:
        mu = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        std[torch.where(std == 0)] = 1
    data = (data - mu) / std
    return data, mu, std


def de_normalization(data, mu, std):
    data = data*std+mu
    return data


def generate_targets(targets, pre_len, tar_len, args):
    """targets output is a (len(targets)-pre_len-tar_len, tar_len, target_features)"""
    len_feat = len(targets)
    output = torch.empty((len(targets)-pre_len-tar_len, tar_len, targets.shape[-1]))
    for i in range(len_feat-pre_len-tar_len):
        output[i] = targets[i+pre_len-1:i+pre_len+tar_len-1]
    return output


def use_cuda(data_batch, use_cuda):
    if use_cuda:
        return data_batch.cuda()
    else:
        return data_batch


def data_loader_1q(features, targets, batch_size, args):
    """features:(len(times), num_nodes, node_feature)
    targets: (len(features)-pre_len-tar_len, tar_len, target_features)
    retrun: (data, target)
    data_batch:(pre_len, batch_size, num_nodes, node_feature)
    target_batch:(tar_len, batch_size, target_features)"""

    orders = torch.tensor(range(len(features) - args.pre_len - args.tar_len))
    if args.shuffle:
        orders = torch.randperm(len(features) - args.pre_len - args.tar_len)

    for i in range(0, len(orders), batch_size):
        if args.file_head == 'Seq2Seq':

            orders_chunk = orders[i: min(i + batch_size, len(orders))]

            data_batch = torch.zeros(args.pre_len, len(orders_chunk), args.node_features)
            observation_data_batch = torch.zeros(args.tar_len, len(orders_chunk), args.node_features)

            target_batch = torch.zeros(args.pre_len + args.tar_len, len(orders_chunk), targets.shape[-1])
            observation_target_batch = torch.zeros(args.tar_len, len(orders_chunk), 1)

            for j, id in enumerate(orders_chunk):
                data_batch[0:args.pre_len, j, 0:-1] = features[id:id + args.pre_len, 0:-1].float()
                data_batch[:, j, -1] = targets[id:id + args.pre_len, 0].float()

                observation_data_batch[:, j, 0:-1] = features[id + args.pre_len:id + args.pre_len+args.tar_len, 0:-1].float()
                observation_data_batch[:, j, -1] = targets[id + args.pre_len:id + args.pre_len+args.tar_len, 0].float()

                target_batch[:, j, :] = targets[id:id + args.pre_len+args.tar_len]
                observation_target_batch[:, j, :] = targets[id + args.pre_len:id + args.pre_len+args.tar_len]

            yield use_cuda(data_batch, args.cuda), use_cuda(observation_data_batch, args.cuda), None, None, \
                  use_cuda(target_batch, args.cuda), use_cuda(observation_target_batch, args.cuda)

        elif args.file_head == 'RNN':
            orders_chunk = orders[i: min(i + batch_size, len(orders))]
            target_batch = torch.empty(args.tar_len, len(orders_chunk), targets.shape[-1])
            orders_chunk = orders[i: i + batch_size]

            data_batch = torch.zeros(args.pre_len, len(orders_chunk), args.node_features)
            observation_data_batch = torch.zeros(args.tar_len, len(orders_chunk), args.node_features)

            for j, id in enumerate(orders_chunk):
                data_batch[:, j, 0:-1] = features[id:id + args.pre_len, 0:-1].float()
                data_batch[:, j, -1] = targets[id:id + args.pre_len, 0].float()

                observation_data_batch[:, j, 0:-1] = features[id + args.pre_len:id + args.pre_len+args.tar_len, 0:-1].float()
                observation_data_batch[:, j, -1] = targets[id + args.pre_len:id + args.pre_len + args.tar_len, 0].float()

                target_batch[:, j, :] = targets[id + args.pre_len:id + args.pre_len + args.tar_len]


            yield use_cuda(data_batch, args.cuda), use_cuda(observation_data_batch, args.cuda), None, None, \
                  use_cuda(target_batch, args.cuda), None


        elif args.file_head == 'graphSeq2Seq':
            orders_chunk = orders[i: min(i + batch_size, len(orders))]
            features_batch = [[] for _ in range(args.pre_len)]
            observation_features_batch = [[] for _ in range(args.tar_len)]
            data_batch = []
            observation_data_batch = []
            if 'graph' in args.file_head:
                for k in range(args.pre_len):
                    for j, id in enumerate(orders_chunk):
                        x = use_cuda(features[id + k, :, 0:-1], args.cuda)
                        edge_index = use_cuda(args.edge_index, args.cuda)
                        d = Data(x=x, edge_index=edge_index)
                        features_batch[k].append(d)

                for k in range(args.tar_len):
                    for j, id in enumerate(orders_chunk):
                        x = use_cuda(features[id + args.pre_len+k, :, 0:-1], args.cuda)
                        edge_index = use_cuda(args.edge_index, args.cuda)
                        d = Data(x=x, edge_index=edge_index)
                        observation_features_batch[k].append(d)

                for k in range(args.pre_len):
                    data_batch.append(Batch.from_data_list(features_batch[k]))

                for k in range(args.tar_len):
                    observation_data_batch.append(Batch.from_data_list(observation_features_batch[k]))

            target_batch = torch.zeros(args.tar_len+args.pre_len, len(orders_chunk), targets.shape[-1])
            observation_target_batch = torch.zeros(args.tar_len, len(orders_chunk), 1)

            adj_neb_batch = torch.zeros(args.pre_len, len(orders_chunk), 1, args.num_nodes)
            observation_adj_neb_batch = torch.zeros(args.tar_len, len(orders_chunk), 1, args.num_nodes)

            adj_batch = torch.zeros(args.pre_len, len(orders_chunk), 2).long()
            observation_adj_batch = torch.zeros(args.tar_len, len(orders_chunk), 2).long()

            for j, id in enumerate(orders_chunk):
                feat_index = np.arange(id, id + args.pre_len)
                target_index = np.arange(id, id+args.pre_len+args.tar_len)
                observ_index = np.arange(id + args.pre_len, id + args.pre_len + args.tar_len)

                target_batch[:, j, :] = targets[target_index]
                observation_target_batch[:, j, :] = targets[observ_index, :]

                q_id = features[feat_index, 0, -1].long()
                adj_batch[:, j, 0] = torch.tensor(j).long()
                adj_batch[:, j, 1] = q_id
                adj_neb_batch[:, j, 0, :] = args.adj[q_id[:], :]

                q_id = features[observ_index, 0, -1].long()
                observation_adj_batch[:, j, 0] = torch.tensor(j).long()
                observation_adj_batch[:, j, 1] = q_id
                observation_adj_neb_batch[:, j, 0, :] = args.adj[q_id[:], :]

            yield data_batch, observation_data_batch, use_cuda(
                adj_batch, args.cuda), use_cuda(observation_adj_batch, args.cuda), \
                  use_cuda(target_batch, args.cuda), use_cuda(observation_target_batch, args.cuda)


def data_loader(features, targets, batch_size, args):
    """
    :param features: a list of features_1q
    :param targets: a list of targets_1q
    :param batch_size:
    :param args:
    :return:
    """
    q_orders = torch.randperm(args.num_nodes)
    for q in q_orders:
        for out in data_loader_1q(features[q], targets[q], batch_size, args):
            yield out


def extend_features(features, targets, q_num, current_adj_feat=None):
    """
    This function is used to extend the log into and adj features
    :param features: (len, feature_len)
    :param targets: (len, target_len)
    :param q_num: (numof queues)
    :return: features_extend: (len adj_shape, feature)
    """
    data_len, feat_len = features.shape
    features_extend = torch.zeros(data_len, q_num, feat_len+1)

    if current_adj_feat is None:
        current_adj_feat = torch.zeros(q_num, feat_len+1)

    for i in range(data_len):
        q_id = features[i, -1].long()
        current_adj_feat[q_id, 0:-2] = features[i, 0:-1]
        current_adj_feat[q_id, -2] = targets[i, 0]
        current_adj_feat[:, -1] = q_id
        features_extend[i] = current_adj_feat

    return features_extend, current_adj_feat


def features_resort(features, targets, q_num):
    '''
    This function is used to re-sort the features and targets according to the queue id and for each
    queue, it is sorted by arrival time.
    :param features:
    :param targets:
    :param q_num:
    :return:
    '''
    features_q = [[] for _ in range(q_num)]
    targets_q = [[] for _ in range(q_num)]
    for i, (feat, target) in enumerate(zip(features, targets)):
        q = int(feat.view(-1)[-1])
        features_q[q].append(feat)
        targets_q[q].append(target)
    features = []
    targets = []
    for features_1q, targets_1q in zip(features_q, targets_q):
        features.append(torch.stack(features_1q))
        targets.append(torch.stack(targets_1q))

    return features, targets



def data_init(args):
    adj, edge_index, features_train, features_val, features_test, \
        targets_train, targets_val, targets_test = load_dataset(path="./simul_data")

    if args.cuda:
        adj = adj.cuda()
        edge_index = edge_index.cuda()
    if len(targets_train.shape) == 1:
        targets_train = targets_train.unsqueeze(1)
        targets_val = targets_val.unsqueeze(1)
        targets_test = targets_test.unsqueeze(1)
    print("featues_train shape: ", features_train.shape)
    print("targets_train shape: ", targets_train.shape)
    print("adj shape: ", adj.shape)

    args.num_nodes = adj.shape[0]
    args.node_features = features_train.shape[-1]
    args.target_features = targets_train.shape[-1]

    q_num = args.num_nodes
    if 'graph' in args.file_head:
        features_train, current_adj_feat = extend_features(features_train, targets_train, q_num)
        features_val, current_adj_feat = extend_features(features_val, targets_val, q_num, current_adj_feat)
        features_test, current_adj_feat = extend_features(features_test, targets_test, q_num, current_adj_feat)

    features_train, targets_train = features_resort(features_train, targets_train, q_num)
    features_val, targets_val = features_resort(features_val, targets_val, q_num)
    features_test, targets_test = features_resort(features_test, targets_test, q_num)

    return adj, edge_index, features_train, features_val, features_test, \
           targets_train, targets_val, targets_test


def debug(features_test, targets_test, args):
    """debug"""
    test_loader_debug = data_loader(features_test, targets_test, args.batch_size, args)
    targets_test = []
    for i, (data_batch, target_batch) in enumerate(test_loader_debug):
        targets_test.extend(target_batch.view(-1))
    visulization(targets_test, [])



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


def resume_checkpoit(args, model, optimizer, resume=True):
    path = os.path.join("./results", args.dataset)
    if not os.path.exists(path):
        os.system('mkdir ' + path)
    args.filename = os.path.join(path, args.file_head) + args.loss + '.pth.tar'
    if resume:
        if os.path.isfile(args.filename):
            print("=> loading checkpoint '{}'".format(args.filename))
            checkpoint = torch.load(args.filename)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.filename))


def rmse(y_pre, y_true):
    inputs = y_pre - y_true
    return (torch.norm(inputs, p=2)/math.sqrt(inputs.shape[0])).item()


def mae(y_pre, y_true):
    inputs = y_pre - y_true
    return (torch.norm(inputs, p=1)/inputs.shape[0]).item()


def mare(y_pre, y_true):
    inputs = torch.abs(y_pre - y_true)
    y_true[torch.where(y_true == 0)] = 1
    inputs = inputs/torch.abs(y_true)
    return (torch.sum(inputs)/inputs.shape[0]).item()


def medae(y_pre, y_true):
    inputs = torch.abs(y_pre - y_true)
    return torch.median(inputs).item()


def mad(y_pre, y_true):
    inputs = y_pre-y_true
    inputs = torch.abs(inputs - torch.median(inputs))
    return torch.median(inputs).item()


def R2(y_pre, y_true):
    mu = torch.mean(y_true)
    tmp = torch.pow(y_pre-y_true, 2).sum()/torch.pow(y_true-mu, 2).sum()
    return 1-tmp.item()

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = 1
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[0].view(-1).float().sum(0, keepdim=True)

    return 1-correct_k.mul_(1 / batch_size).squeeze()


def evaluation(target, prediction, method='rmse'):
    target = target.view(-1, 1)
    prediction = prediction.view(-1, 1)
    with torch.no_grad():
        if method == 'roc':
            fpr, tpr, thresholds = metrics.roc_curve(target, prediction, pos_label=2)
            return metrics.auc(fpr, tpr)
        elif method == 'rmse':
            return rmse(prediction, target)
        elif method == 'mae':
            return mae(prediction, target)
        elif method == 'medae':
            return medae(prediction, target)
        elif method == 'mad':
            return mad(prediction, target)
        elif method == 'R2':
            return R2(prediction, target)
        elif method == 'accu':
            return accuracy(prediction, target)


def visulization(target, prediction, path='./', ratio=0.001, show=False, title=None):
    data = {'target': np.array(target), 'prediction': np.array(prediction)}
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(data, f)
    target_index = np.arange(1, len(target), int(1 / ratio))
    prediction_index = np.arange(1, len(prediction), int(1 / ratio))
    if show:
        import matplotlib.pyplot as plt
        plt.plot(np.array(target)[target_index], 'r^-')
        plt.plot(np.array(prediction)[prediction_index], 'b')
        plt.savefig(path+'.png')
        plt.title(title)
        plt.show()




if __name__ == '__main__':
    file_head = 'graph_VAE_GRU'
    epoch = 5
    path = './logs/train/'+file_head+'train_epoch_'+str(epoch)
    with open(path+'.pkl', 'rb') as f:
        data = pickle.load(f)

    import matplotlib.pyplot as plt
    plt.plot(data['target'], 'r^-')
    plt.plot(data['prediction'], 'b')
    plt.savefig(path+'.png')
    plt.show()

    path = './logs/val/'+file_head+'val_epoch_'+str(epoch)
    with open(path+'.pkl', 'rb') as f:
        data = pickle.load(f)

    plt.plot(data['target'], 'r^-')
    plt.plot(data['prediction'], 'b')
    plt.savefig(path+'.png')
    plt.show()