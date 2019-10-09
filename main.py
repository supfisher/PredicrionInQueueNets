from model import *
from utils import *
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--pre_len', type=int, default=5,
                    help='the length of input data sequence.')
parser.add_argument('--tar_len', type=int, default=3,
                    help='the length of output target sequence.')
parser.add_argument('--batch_size', type=int, default=4,
                    help='the batch size.')
parser.add_argument('--feq', type=int, default=30,
                    help='frequency to show the accuracy.')
parser.add_argument('--mem_efficient', action='store_true', default=False,
                    help='whether to use a membory efficient mode')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train(train_loader, model, criterion, optimizer, epoch, mode='train'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        int(len(features_train)/args.batch_size),
        [batch_time, data_time, losses],
        prefix=mode+"_Epoch: [{}]".format(epoch))

    t = time.time()
    model.train()
    for i, (data, target) in enumerate(train_loader):
        output, hn = model(data)
        output = output[-args.tar_len:].squeeze()
        loss_train = criterion(output, target)
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = get_losses(target.view(-1).detach(), output.view(-1).detach(), method='rmse')
        losses.update(loss.item(), args.batch_size)
        if i % args.feq == 0:
            progress.display(i)


def test(test_loader, model, criterion, epoch=0, mode='test'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        int(len(features_test)/args.batch_size),
        [batch_time, data_time, losses],
        prefix=mode+"_Epoch: [{}]".format(epoch))

    t = time.time()
    model.eval()
    for i, (data, target) in enumerate(test_loader):
        output, hn = model(data)
        output = output[-args.tar_len:].squeeze()
        loss_test = criterion(output, target)
        loss = get_losses(target.view(-1).detach(), output.view(-1).detach(), method='rmse')
        losses.update(loss.item(), args.batch_size)
    progress.display(i)


def data_init(args):
    adj, features = load_path()
    adj = torch.tensor(adj).float()
    features_train, features_val, features_test = train_test_split(features, ratio=(0.5, 0.3,))
    features_train, features_val, features_test = torch.tensor(features_train).float(), torch.tensor(
        features_val).float(), torch.tensor(features_test).float()
    print("featues_train shape: ", features_train.shape)
    args.num_nodes = features_train.shape[1]
    assert adj.shape[0] == features_train.shape[1]
    args.node_features = features_train.shape[2]

    targets_train = generate_targets(features_train, args.pre_len, args.tar_len, args)
    targets_val = generate_targets(features_val, args.pre_len, args.tar_len, args)
    targets_test = generate_targets(features_val, args.pre_len, args.tar_len, args)
    return adj, features_train, features_val, features_test, targets_train, targets_val, targets_test



if __name__=="__main__":
    # Load data
    adj, features_train, features_val, features_test, targets_train, targets_val, targets_test = data_init(args)
    train_loader, val_loader, test_loader = None, None, None
    # Model and optimizer
    model = TGCN(in_feat=args.node_features,
                 out_feat=args.num_nodes,
                 G_feat=1,
                 n_layers=3,
                 dropout=0.1,
                 adj=adj,
                 mode='GRU')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        print("use cuda")

    t_total = time.time()
    if not args.mem_efficient:
        train_loader = [_ for _ in data_loader(features_train, targets_train, args.batch_size, args)]
        val_loader = [_ for _ in data_loader(features_val, targets_val, args.batch_size, args)]
    for epoch in range(args.epochs):
        if args.mem_efficient:
            train_loader = data_loader(features_test, targets_test, args.batch_size, args)
            val_loader = data_loader(features_val, targets_val, args.batch_size, args)
        train(train_loader, model, criterion, optimizer, epoch)
        test(val_loader, model, criterion, epoch, mode='val')

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    test_loader = data_loader(features_test, targets_test, args.batch_size, args)
    test(test_loader, model, criterion)

