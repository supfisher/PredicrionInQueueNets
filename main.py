from model import *
from utils import *
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='Whether to shuffle the dataset.')
parser.add_argument('--graphlib', action='store_true', default=False,
                    help='Whether to use the torch_geometric graph lib.')
parser.add_argument('--padding', action='store_true', default=True,
                    help='Whether to padding the input. If yes, the seq_len of RNN model==pre_len_tar_len')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=4,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--pre_len', type=int, default=1,
                    help='the length of input data sequence.')
parser.add_argument('--tar_len', type=int, default=1,
                    help='the length of output target sequence.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='the batch size.')
parser.add_argument('--feq', type=int, default=30,
                    help='frequency to show the accuracy.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print("use cuda")


def train(train_loader, model, criterion, optimizer, epoch, mode='train'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.2f')
    train_losses = AverageMeter('Training Loss', ':.4f')
    progress = ProgressMeter(
        100,
        [batch_time, data_time, losses, train_losses],
        prefix=mode+"_Epoch: [{}]".format(epoch))

    t = time.time()
    model.train()
    for i, (data, target) in enumerate(train_loader):
        output, hn = model(data)
        output = output[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        target = target[-args.tar_len:].reshape(-1).long()
        loss_train = criterion(output, target)

        loss_train.backward()
        # for name, params in model.named_parameters():
        #     print(name, ": grad: ", params.grad.data)
        #     print(name, ": data: ", params.data)
        optimizer.step()
        optimizer.zero_grad()

        loss = get_losses(target, output, method='rmse')
        losses.update(loss, output.shape[0])
        train_losses.update(loss_train.item(), args.batch_size)
        if i % args.feq == 0:
            progress.display(i)


def test(test_loader, model, criterion, epoch=0, mode='test'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    test_losses = AverageMeter(mode+' Loss', ':.4f')
    progress = ProgressMeter(
        100,
        [batch_time, data_time, losses, test_losses],
        prefix=mode+"_Epoch: [{}]".format(epoch))

    t = time.time()
    model.eval()
    targets = []
    predictions = []
    for i, (data, target) in enumerate(test_loader):
        output, hn = model(data)
        output = output[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        target = target[-args.tar_len:].squeeze().long()
        loss_test = criterion(output, target)
        test_losses.update(loss_test.item(), args.batch_size)
        loss = get_losses(target, output, method='rmse')
        losses.update(loss, 1)

        targets.extend(target.view(-1).detach())
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        predictions.extend(pred.view(-1).detach())

    progress.display(i)
    if mode == 'test':
        print("targets: ", targets)
        print("predictions: ", predictions)
        visulization(targets, predictions)


def data_init(args):
    adj, edge_index, features, targets = load_path()
    adj = torch.tensor(adj).float()

    features_train, features_val, features_test, targets_train, targets_val, targets_test\
        = train_test_split(features, targets, ratio=(0.5, 0.3,))
    features_train, features_val, features_test = torch.tensor(features_train).float(), torch.tensor(
        features_val).float(), torch.tensor(features_test).float()
    targets_train, targets_val, targets_test = torch.tensor(targets_train).float(), torch.tensor(
        targets_val).float(), torch.tensor(targets_test).float()

    if len(targets_train.shape) == 1:
        targets_train = targets_train.unsqueeze(1)
        targets_val = targets_val.unsqueeze(1)
        targets_test = targets_test.unsqueeze(1)


    features_train, mu, std = normalization(features_train)
    features_val, _, _ = normalization(features_val, mu, std)
    features_test, _, _ = normalization(features_test, mu, std)

    print("featues_train shape: ", features_train.shape)
    print("targets_train shape: ", targets_train.shape)
    args.num_nodes = features_train.shape[1]
    assert adj.shape[0] == features_train.shape[1]
    args.node_features = features_train.shape[2]

    return adj, edge_index, features_train, features_val, features_test, targets_train, targets_val, targets_test

def debug(features_test, targets_test):
    """debug"""
    test_loader_debug = data_loader(features_test, targets_test, args.batch_size, args)
    targets_test = []
    for i, (data_batch, target_batch) in enumerate(test_loader_debug):
        targets_test.extend(target_batch.view(-1))
    visulization(targets_test, [])


if __name__ == "__main__":
    # Load data
    adj, edge_index, features_train, features_val, features_test, targets_train, targets_val, targets_test = data_init(args)
    args.edge_index = edge_index
    train_loader, val_loader, test_loader = None, None, None
    # debug(features_test, targets_test)
    # debug(features_val, targets_val)
    # Model and optimizer
    # model = TGCN(in_feat=args.node_features,
    #              out_feat=targets_train.shape[-1],
    #              G_hidden=1,
    #              RNN_hidden=3,
    #              seq_len=args.pre_len,
    #              n_layers=2,
    #              dropout=args.dropout,
    #              adj=adj,
    #              mode='GRU')
    model = RNN(in_feat=args.node_features*adj.shape[0], out_feat=5, n_layers=2, dropout=args.dropout, mode='GRU')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        print("use cuda")

    t_total = time.time()
    for epoch in range(args.epochs):
        train_loader = data_loader(features_train, targets_train, args.batch_size, args)
        val_loader = data_loader(features_val, targets_val, args.batch_size, args)
        train(train_loader, model, criterion, optimizer, epoch)
        test(val_loader, model, criterion, epoch, mode='val')

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    test_loader = data_loader(features_test, targets_test, args.batch_size, args)
    test(test_loader, model, criterion)

