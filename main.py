from model import *
from utils import *
import time
import argparse
import numpy as np

import torch
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
parser.add_argument('--lr', type=float, default=5,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--pre_len', type=int, default=15,
                    help='the length of input data sequence.')
parser.add_argument('--tar_len', type=int, default=1,
                    help='the length of output target sequence.')
parser.add_argument('--batch_size', type=int, default=4,
                    help='the batch size.')
parser.add_argument('--feq', type=int, default=30,
                    help='frequency to show the accuracy.')

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
    train_losses = AverageMeter('Training Loss', ':.4e')
    progress = ProgressMeter(
        100,
        [batch_time, data_time, losses, train_losses],
        prefix=mode+"_Epoch: [{}]".format(epoch))

    t = time.time()
    model.train()
    for i, (data, target) in enumerate(train_loader):
        output, hn = model(data)
        output = output[-args.tar_len:].squeeze()
        loss_train = criterion(output.reshape(args.batch_size,-1), target.reshape(args.batch_size,-1))
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        loss = get_losses(target.view(-1).detach(), output.view(-1).detach(), method='rmse')
        losses.update(loss, 1)
        train_losses.update(loss_train.item(), args.batch_size)
        if i % args.feq == 0:
            print("target: ", target.view(-1).detach())
            print("prediction: ", output.view(-1).detach())
            progress.display(i)


def test(test_loader, model, criterion, epoch=0, mode='test'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    test_losses = AverageMeter(mode+' Loss', ':.4e')
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
        output = output[-args.tar_len:].squeeze()
        loss_test = criterion(output, target)
        loss = get_losses(target.view(-1).detach(), output.view(-1).detach(), method='rmse')
        targets.extend(target.view(-1).detach())
        predictions.extend(output.view(-1).detach())
        losses.update(loss, 1)
        test_losses.update(loss_test.item(), args.batch_size)
    progress.display(i)
    if mode == 'test':
        print("targets: ", targets)
        print("predictions: ", predictions)
        visulization(targets, predictions)


def data_init(args):
    adj, features, targets = load_path()
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

    targets_train,_,_ = nomalization(targets_train)
    targets_val,_,_ = nomalization(targets_val)
    targets_test,_,_ = nomalization(targets_test)
    print("featues_train shape: ", features_train.shape)
    print("targets_train shape: ", targets_train.shape)
    args.num_nodes = features_train.shape[1]
    assert adj.shape[0] == features_train.shape[1]
    args.node_features = features_train.shape[2]

    targets_train = generate_targets(targets_train, args.pre_len, args.tar_len, args)
    targets_val = generate_targets(targets_val, args.pre_len, args.tar_len, args)
    targets_test = generate_targets(targets_test, args.pre_len, args.tar_len, args)


    return adj, features_train, features_val, features_test, targets_train, targets_val, targets_test



if __name__=="__main__":
    # Load data
    adj, features_train, features_val, features_test, targets_train, targets_val, targets_test = data_init(args)
    train_loader, val_loader, test_loader = None, None, None
    # Model and optimizer
    model = TGCN(in_feat=args.node_features,
                 out_feat=targets_train.shape[-1],
                 G_feat=1,
                 n_layers=3,
                 dropout=0.1,
                 adj=adj,
                 mode='GRU')
    criterion = nn.KLDivLoss()
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

