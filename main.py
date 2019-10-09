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
parser.add_argument('--epochs', type=int, default=1,
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
parser.add_argument('--batch_size', type=int, default=1,
                    help='the batch size.')
parser.add_argument('--feq', type=int, default=30,
                    help='frequency to show the accuracy.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features = load_data()
adj = torch.tensor(adj).float()
features_train, features_val, features_test = train_test_split(features, ratio=(0.5, 0.3, ))
features_train, features_val, features_test = torch.tensor(features_train).float(), torch.tensor(features_val).float(), torch.tensor(features_test).float()
print("featues_train shape: ", features_train.shape)
args.num_nodes = features_train.shape[1]
assert adj.shape[0] == features_train.shape[1]
args.num_features = features_train.shape[2]

def generate_targets(features, pre_len, tar_len):
    """for the test of the demo, we use the data from the features as target
    targets output is a (len(features)-pre_len-tar_len, tar_len, args.num_features)"""
    len_feat = len(features)
    output = torch.empty((len(features)-pre_len-tar_len, tar_len, args.num_nodes))
    for i in range(len_feat-pre_len-tar_len):
        output[i,:,:] = features[i+pre_len:i+pre_len+tar_len,:,0].squeeze()
    return output


# Model and optimizer
# model = GCN(nfeat=num_features,
#             nhid=args.hidden,
#             nout=1,
#             dropout=args.dropout,
#             adj=adj)
model = TGCN(nfeat=args.num_features,
             nhid=args.num_nodes,
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


def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(features_train),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    t = time.time()
    model.train()
    optimizer.zero_grad()
    train_orders = torch.randperm(len(features_train)-args.pre_len-args.tar_len)
    targets = generate_targets(features_train, args.pre_len, args.tar_len)
    outputs = torch.empty(targets.shape)
    for i, order in enumerate(train_orders):
        target = targets[order]
        output, hn = model(features_train[order:order+args.pre_len,:,:])
        output = output[-args.tar_len:].squeeze()
        loss_train = criterion(output, target)
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()
        outputs[order] = output
        loss = get_losses(target.view(-1).detach(), output.view(-1).detach(), method='rmse')
        losses.update(loss.item(), args.batch_size)
        if i % args.feq == 0:
            progress.display(i)
    visulization(targets.view(-1).detach(), outputs.view(-1).detach())


def test():
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(features_test),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    t = time.time()
    model.eval()
    optimizer.zero_grad()
    test_orders = torch.randperm(len(features_test)-args.pre_len-args.tar_len)
    targets = generate_targets(features_test, args.pre_len, args.tar_len)
    outputs = torch.empty(targets.shape)
    for i, order in enumerate(test_orders):
        target = targets[order]
        output, hn = model(features_test[order:order+args.pre_len,:,:])
        output = output[-args.tar_len:].squeeze()
        loss_test = criterion(output, target)
        outputs[order] = output
    loss = get_losses(targets.view(-1).detach(), outputs.view(-1).detach(), method='rmse')
    losses.update(loss.item(), args.batch_size)
    progress.display(i)
    visulization(targets.view(-1).detach(), outputs.view(-1).detach())






# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
test()

