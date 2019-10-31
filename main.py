from model import *
from utils import *
import time
import argparse
import numpy as np
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--shuffle', action='store_true', default=True,
                    help='Whether to shuffle the dataset.')
parser.add_argument('--graphlib', action='store_true', default=False,
                    help='Whether to use the torch_geometric graph lib.')
parser.add_argument('--padding', action='store_true', default=False,
                    help='Whether to padding the input. If yes, the seq_len of RNN model==pre_len_tar_len')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=120,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--pre_len', type=int, default=5,
                    help='the length of input data sequence.')
parser.add_argument('--tar_len', type=int, default=1,
                    help='the length of output target sequence.')
parser.add_argument('--batch_size', type=int, default=1200,
                    help='the batch size.')
parser.add_argument('--feq', type=int, default=30,
                    help='frequency to show the accuracy.')





""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

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
        output, hn = model(data)
        output = output[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        target = target[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        loss_train = criterion(output, target)

        loss_train.backward()
        # for name, params in model.named_parameters():
        #     print(name, ": grad: ", params.grad.data)
        #     print(name, ": data: ", params.data)
        average_gradients(model)
        optimizer.step()
        optimizer.zero_grad()

        loss = get_losses(target, output, method='rmse')
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
    for i, (data, target) in enumerate(test_loader):
        output, hn = model(data)
        output = output[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        target = target[-args.tar_len:].reshape(args.batch_size*args.tar_len, -1)
        loss_test = criterion(output, target)
        test_losses.update(loss_test.item(), args.batch_size)
        loss = get_losses(target, output, method='rmse')
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

    targets_train, mu, std = normalization(targets_train)
    targets_val, _, _ = normalization(targets_val, mu, std)
    targets_test, _, _ = normalization(targets_test, mu, std)

    features_train, mu, std = normalization(features_train)
    features_val, _, _ = normalization(features_val, mu, std)
    features_test, _, _ = normalization(features_test, mu, std)

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



def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function

        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


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
        model = TGCN(in_feat=args.node_features,
                     out_feat=args.node_features,
                     G_hidden=3,
                     seq_len=seq_len,
                     n_layers=2,
                     dropout=args.dropout,
                     adj=adj,
                     mode='GRU')
    else:
        model = RNN(in_feat=args.node_features * adj.shape[0], out_feat=args.node_features * adj.shape[0], n_layers=2,
                    dropout=args.dropout, mode='GRU')

    model.cuda()


    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    t_total = time.time()
    for epoch in range(args.epochs):
        train_loader = data_loader(features_train, targets_train, args.batch_size, args)
        val_loader = data_loader(features_val, targets_val, args.batch_size, args)
        train(train_loader, model, criterion, optimizer, epoch)
        test(val_loader, model, criterion, epoch, mode='val')

    if args.rank == 0:
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        test_loader = data_loader(features_test, targets_test, args.batch_size, args)
        test(test_loader, model, criterion)
        os.system('spd-say "your program is finished"')


if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print("use cuda")
    if args.graphlib:
        args.file_head = 'graph_GRU_'
    else:
        args.file_head = 'GRU_'
    main(args)
