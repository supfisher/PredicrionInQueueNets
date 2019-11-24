from models import *
from utils import *
from pytools import *
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
parser.add_argument('--file_head', type=str, default='RNN',
                    help='which regression model to use.')
parser.add_argument('--show', action='store_true', default=False,
                    help='whether to show the result.')
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to load the stored model')
parser.add_argument('--early_stop', action='store_true', default=True,
                    help='whether to stop training early')
parser.add_argument('--loss', type=str, default='l1',
                    help='which loss function to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='the start epoch.')
parser.add_argument('--epochs', type=int, default=80,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum.')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--pre_len', type=int, default=10,
                    help='the length of input data sequence.')
parser.add_argument('--tar_len', type=int, default=3,
                    help='the length of output target sequence.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='training batch size.')
parser.add_argument('--val_batch_size', type=int, default=256,
                    help='validation and test batch size.')
parser.add_argument('--feq', type=int, default=30,
                    help='frequency to show the accuracy.')


def forward(data, observ_data, adj1, observ_adj1, target, observ_target, criterion, model, observ=True):
    if args.file_head == 'Seq2Seq':
        output, shift_out, shift_in = model(data, observ_data, observ_target, observ=observ)
        # output_ = output.reshape(-1, 1)
        # target_ = target.reshape(-1, 1)
        output_ = output[-args.tar_len:].reshape(-1, 1)
        target_ = target[-args.tar_len:].reshape(-1, 1)
        loss = criterion(output_, target_, shift_out, shift_in, args.loss)
        return output, loss


    elif args.file_head == 'graphSeq2Seq':
        output, shift_out, shift_in = model(data, observ_data, adj1, observ_adj1, observ=observ)
        # output_ = output.reshape(-1, 1)
        # target_ = target.reshape(-1, 1)
        output_ = output[-args.tar_len:].reshape(-1, 1)
        target_ = target[-args.tar_len:].reshape(-1, 1)
        loss = criterion(output_, target_, shift_out, shift_in, args.loss)
        return output, loss

    elif args.file_head == 'RNN':
        output = model(data)
        # output_ = output.reshape(-1, 1)
        target_ = target.reshape(-1, 1)
        output_ = output[-args.tar_len:].reshape(-1, 1)
        # target_ = target[-args.tar_len:].reshape(-1, 1)
        loss = criterion(output_, target_, args.loss)
        return output, loss


def train(train_loader, model, criterion, optimizer, epoch, mode='train', args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    rmse = AverageMeter('rmse', ':6.5f')
    mae = AverageMeter('mae', ':6.5f')
    train_losses = AverageMeter('Training Loss', ':.5f')
    progress = ProgressMeter(
        100,
        [batch_time, data_time, train_losses, rmse, mae],
        prefix=mode + "_Epoch: [{}]".format(epoch))

    t = time.time()
    model.train()
    targets = []
    predictions = []
    adj = args.adj
    for i, (data, observ_data, adj1, observ_adj1, target, observ_target) in enumerate(train_loader):
        output, loss_train = forward(data, observ_data, adj1, observ_adj1,
                                     target, observ_target, criterion, model, observ=False)

        loss_train.backward()

        optimizer.step()

        optimizer.zero_grad()

        target = target[-args.tar_len:].view(-1)
        output = output[-args.tar_len:].view(-1)

        rmse.update(evaluation(target, output, method='rmse'), output.shape[0])
        mae.update(evaluation(target, output, method='mae'), output.shape[0])

        train_losses.update(loss_train.item(), output.shape[0])

        targets.extend(target.detach())
        predictions.extend(output.detach())

        if i % args.feq == 0 and args.rank == 0:
            args.writer.add_scalar(args.file_head + mode + 'rmse', rmse.avg, args.feq * args.train_iter)
            args.writer.add_scalar(args.file_head + mode + 'mae', mae.avg, args.feq * args.train_iter)

            args.writer.add_scalar(args.file_head + mode + 'training loss', train_losses.avg, args.feq * args.train_iter)
            args.train_iter += 1
            progress.display(i)

    print('train_epoch: ', epoch, 'rmse',
          evaluation(torch.stack(targets), torch.stack(predictions), method='rmse'))
    print('train_epoch: ', epoch, 'mae',
          evaluation(torch.stack(targets), torch.stack(predictions), method='mae'))
    print('train_epoch: ', epoch, 'mad',
          evaluation(torch.stack(targets), torch.stack(predictions), method='mad'))
    print('train_epoch: ', epoch, 'R2',
          evaluation(torch.stack(targets), torch.stack(predictions), method='R2'))

    if epoch % 1 == 0 and args.rank == 0:
        dir = os.path.join('./logs', mode)
        if not os.path.isdir(dir):
            os.system('mkdir ' + dir)
        path = os.path.join(dir, args.file_head + mode + '_epoch_' + str(epoch))
        visulization(targets, predictions, path=path, ratio=0.1, show=args.show, title=mode + '_epoch: ' + str(epoch))


def test(test_loader, model, criterion, epoch=0, mode='test', args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    rmse = AverageMeter('rmse', ':6.5f')
    mae = AverageMeter('mae', ':6.5f')
    test_losses = AverageMeter(mode + ' Loss', ':.5f')
    progress = ProgressMeter(
        100,
        [batch_time, data_time, test_losses, rmse, mae],
        prefix=mode + "_Epoch: [{}]".format(epoch))

    t = time.time()
    model.eval()
    targets = []
    predictions = []
    i = 0
    adj = args.adj
    for i, (data, observ_data, adj1, observ_adj1, target, observ_target) in enumerate(test_loader):

        output, loss_test = forward(data, observ_data, adj1,
                                    observ_adj1, target, observ_target, criterion, model, observ=False)

        target = target[-args.tar_len:].view(-1)
        output = output[-args.tar_len:].view(-1)

        rmse.update(evaluation(target, output, method='rmse'), output.shape[0])
        mae.update(evaluation(target, output, method='mae'), output.shape[0])

        test_losses.update(loss_test.item(), output.shape[0])

        targets.extend(target.detach())
        predictions.extend(output.detach())

        if mode == 'test' and args.rank == 0:
            args.writer.add_scalar(args.file_head + mode + 'rmse', rmse.avg, args.feq * args.train_iter)
            args.writer.add_scalar(args.file_head + mode + 'mae', mae.avg, args.feq * args.train_iter)

            args.writer.add_scalar(args.file_head + mode + 'training loss', test_losses.avg, args.feq * args.test_iter)
            args.test_iter += 1

    if epoch % 1 == 0 and args.rank == 0:
        progress.display(i)
        print('test_epoch: ', epoch, 'rmse',
              evaluation(torch.stack(targets), torch.stack(predictions), method='rmse'))
        print('test_epoch: ', epoch, 'mae',
              evaluation(torch.stack(targets), torch.stack(predictions), method='mae'))
        print('test_epoch: ', epoch, 'mad',
              evaluation(torch.stack(targets), torch.stack(predictions), method='mad'))
        print('test_epoch: ', epoch, 'R2',
              evaluation(torch.stack(targets), torch.stack(predictions), method='R2'))

        dir = os.path.join('./logs', mode)
        if not os.path.isdir(dir):
            os.system('mkdir ' + dir)
        path = os.path.join(dir, args.file_head + mode + '_epoch_' + str(epoch))
        visulization(targets, predictions, path=path, ratio=0.1, show=args.show, title=mode + '_epoch: ' + str(epoch))

    return test_losses.avg



def main_worker(args):
    global best_loss
    if args.rank == 0:
        args.writer = SummaryWriter('./logs')
        args.train_iter = 0
        args.test_iter = 0
    # Load data
    adj, edge_index, features_train, features_val, features_test, \
    targets_train, targets_val, targets_test = data_init(args)
    args.adj = adj
    args.num_nodes = adj.shape[0]
    args.edge_index = edge_index
    args.adj_shape = adj.shape[0]
    train_loader, val_loader, test_loader = None, None, None

    if 'Seq2Seq' == args.file_head:
        model = Seq2Seq(in_dim=args.node_features, hid_dim=4, out_dim=2,
                        n_layers=2, dropout=args.dropout)
        criterion = Seq2SeqLoss()

    elif 'graphSeq2Seq' == args.file_head:
        model = GraphSeq2Seq(in_dim=args.node_features, hid_dim=4, out_dim=2,
                          n_layers=2, adj=adj, dropout=args.dropout)
        criterion = GraphSeq2SeqLoss()
        # input = torch.zeros(args.pre_len, args.batch_size, args.node_features-1)
        # hid = None
        # args.writer.add_graph(model, input_to_model=[input, hid])
    elif 'RNN' == args.file_head:
        model = RNN(in_dim=args.node_features, hid_dim=4,
                          n_layers=2, dropout=args.dropout)
        criterion = RNNLoss()
    model = model.to(args.device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    resume_checkpoit(args, model, optimizer, args.resume)
    # model = torch.nn.DataParallel(model)
    # model = model.cuda()

    t_total = time.time()

    early_stop = EarlyStopping(patience=5, verbose=True, delta=0, path=args.filename)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loader = data_loader(features_train, targets_train, args.batch_size, args)
        val_loader = data_loader(features_val, targets_val, args.val_batch_size, args)
        train(train_loader, model, criterion, optimizer, epoch, args=args)
        loss = test(val_loader, model, criterion, epoch, mode='val', args=args)
        early_stop(loss, model, optimizer, epoch)
        if args.early_stop and early_stop.early_stop:
            print("Early stopping")
            break

    if args.rank == 0:
        resume_checkpoit(args, model, optimizer)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        test_loader = data_loader(features_test, targets_test, args.batch_size, args)
        test(test_loader, model, criterion, args=args)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.cuda = torch.cuda.is_available()
    print("device: ", args.device)
    args.rank = 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.dataset = 'pagerank'

    main_worker(args)
