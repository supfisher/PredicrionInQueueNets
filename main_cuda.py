from models import *
from utils import *
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--shuffle', action='store_true', default=True,
                    help='Whether to shuffle the dataset.')
parser.add_argument('--graphlib', action='store_true', default=True,
                    help='Whether to use the torch_geometric graph lib.')
parser.add_argument('--padding', action='store_true', default=False,
                    help='Whether to padding the input. If yes, the seq_len of RNN model==pre_len_tar_len')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='the start epoch.')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum.')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--pre_len', type=int, default=5,
                    help='the length of input data sequence.')
parser.add_argument('--tar_len', type=int, default=1,
                    help='the length of output target sequence.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='the batch size.')
parser.add_argument('--feq', type=int, default=30,
                    help='frequency to show the accuracy.')


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
    targets = []
    predictions = []
    for i, (data, target) in enumerate(train_loader):
        output, mu, logvar = model(data)
        output = output[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        target = target[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        loss_train = criterion(output, target, mu, logvar)

        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = evaluation(target.cpu(), output.cpu(), method='rmse')
        losses.update(loss, output.shape[0])
        train_losses.update(loss_train.item(), args.batch_size)
        if i % args.feq == 0 and args.rank == 0:
            args.writer.add_scalar(args.file_head+'expriment loss', loss, args.feq*args.train_iter)
            args.writer.add_scalar(args.file_head+'training loss', loss_train.item(), args.feq*args.train_iter)
            args.train_iter += 1
            progress.display(i)

        targets.extend(target.view(-1).detach())
        predictions.extend(output.view(-1).detach())

    if epoch % 5 == 0 and args.rank == 0:
        dir = os.path.join('./logs', mode)
        if not os.path.isdir(dir):
            os.system('mkdir ' + dir)
        path = os.path.join(dir, args.file_head + mode + '_epoch_' + str(epoch))
        visulization(targets, predictions, path=path, ratio=0.05, show=False)


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
    i=0
    for i, (data, target) in enumerate(test_loader):
        output, mu, logvar = model(data)
        output = output[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        target = target[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        loss_test = criterion(output, target, mu, logvar)
        test_losses.update(loss_test.item(), args.batch_size)
        loss = evaluation(target.cpu(), output.cpu(), method='rmse')
        losses.update(loss, 1)

        targets.extend(target.view(-1).detach())
        predictions.extend(output.view(-1).detach())
        
        if mode == 'test' and args.rank == 0:
            args.writer.add_scalar(args.file_head+mode + 'expriment loss', loss, i)
            args.writer.add_scalar(args.file_head+mode + 'training loss', loss_test.item(), args.feq * args.test_iter)
            args.test_iter += 1

    if args.rank == 0:
        progress.display(i)
        dir = os.path.join('./logs', mode)
        if not os.path.isdir(dir):
            os.system('mkdir ' + dir)
        path = os.path.join(dir, args.file_head+mode+'_epoch_' + str(epoch))
        visulization(targets, predictions, path=path, ratio=0.05, show=False)
    return loss

def data_init(args):
    # adj, edge_index, features, targets = load_raw_path(data_split=False)
    # adj = torch.tensor(adj).float()
    #
    # features_train, features_val, features_test, targets_train, targets_val, targets_test\
    #     = train_test_split(features, targets, ratio=(0.5, 0.3,))
    # features_train, features_val, features_test = torch.tensor(features_train).float(), torch.tensor(
    #     features_val).float(), torch.tensor(features_test).float()
    # targets_train, targets_val, targets_test = torch.tensor(targets_train).float(), torch.tensor(
    #     targets_val).float(), torch.tensor(targets_test).float()
    #
    # if len(targets_train.shape) == 1:
    #     targets_train = targets_train.unsqueeze(1)
    #     targets_val = targets_val.unsqueeze(1)
    #     targets_test = targets_test.unsqueeze(1)
    #
    # targets_train, mu, std = normalization(targets_train)
    # targets_val, _, _ = normalization(targets_val, mu, std)
    # targets_test, _, _ = normalization(targets_test, mu, std)
    #
    # features_train, mu, std = normalization(features_train)
    # features_val, _, _ = normalization(features_val, mu, std)
    # features_test, _, _ = normalization(features_test, mu, std)

    adj, edge_index, features_train, features_val, features_test, \
        targets_train, targets_val, targets_test = load_dataset(path="./simul_data")
    if len(targets_train.shape) == 1:
        targets_train = targets_train.unsqueeze(1)
        targets_val = targets_val.unsqueeze(1)
        targets_test = targets_test.unsqueeze(1)
    print("featues_train shape: ", features_train.shape)
    print("targets_train shape: ", targets_train.shape)

    args.num_nodes = features_train.shape[1]
    assert adj.shape[0] == features_train.shape[1]
    args.node_features = features_train.shape[2]

    return adj, edge_index, features_train, features_val, features_test, \
           targets_train, targets_val, targets_test


def debug(features_test, targets_test):
    """debug"""
    test_loader_debug = data_loader(features_test, targets_test, args.batch_size, args)
    targets_test = []
    for i, (data_batch, target_batch) in enumerate(test_loader_debug):
        targets_test.extend(target_batch.view(-1))
    visulization(targets_test, [])


def main_worker(args):
    global best_loss
    best_loss = -100000
    if args.rank == 0:
        args.writer = SummaryWriter('./logs')
        args.train_iter = 0
        args.test_iter = 0
    # Load data
    adj, edge_index, features_train, features_val, features_test, \
    targets_train, targets_val, targets_test = data_init(args)
    args.edge_index = edge_index
    train_loader, val_loader, test_loader = None, None, None
    # debug(features_test, targets_test)
    # debug(features_val, targets_val)
    # Model and optimizer
    if args.padding:
        seq_len = args.pre_len + args.tar_len
    else:
        seq_len = args.pre_len

    if args.graphlib:
        G_model = VAE
        model = TGCN(in_feat=args.node_features,
                     out_feat=args.node_features,
                     G_model=G_model, G_hidden=1, G_latent=1,
                     n_layers=2,
                     dropout=args.dropout,
                     edge_index=edge_index,
                     mode='GRU')
        criterion = MyLoss()
    else:
        model = RNN(in_feat=args.node_features * adj.shape[0], out_feat=args.node_features * adj.shape[0], n_layers=2,
                    dropout=args.dropout, mode='GRU')
        criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    path = os.path.join("./results", args.dataset)
    if not os.path.exists(path):
        os.system('mkdir ' + path)
    args.filename = os.path.join(path, args.file_head)+'.pth.tar'
    if os.path.isfile(args.filename):
        print("=> loading checkpoint '{}'".format(args.filename))
        checkpoint = torch.load(args.filename)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.filename))

    model = torch.nn.DataParallel(model).cuda()

    t_total = time.time()
    is_best = True

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        train_loader = data_loader(features_train, targets_train, args.batch_size, args)
        val_loader = data_loader(features_val, targets_val, args.batch_size, args)
        train(train_loader, model, criterion, optimizer, epoch)
        loss = test(val_loader, model, criterion, epoch, mode='val')
        is_best = loss > best_loss
        best_loss = min(loss, best_loss)
        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=args.filename)

    if args.rank == 0:
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        test_loader = data_loader(features_test, targets_test, args.batch_size, args)
        test(test_loader, model, criterion)


if __name__ == "__main__":
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.cuda = torch.cuda.is_available()
    print("device: ", args.device)
    args.rank = 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.dataset = 'pagerank'
    if args.graphlib:
        args.file_head = 'graph_GRU'
    else:
        args.file_head = 'GRU'
    main_worker(args)
