import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn.conv import SAGEConv as gnn
from .layers import GraphConvolution

class GCNEncoder(nn.Module):
    """Encoder using GCN layers"""

    def __init__(self, n_feat, n_hid, n_latent, dropout):
        super(GCNEncoder, self).__init__()
        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2_mu = GraphConvolution(n_hid, n_latent)
        self.gc2_sig = GraphConvolution(n_hid, n_latent)
        self.dropout = dropout


    def forward(self, x, adj):
        # First layer shared between mu/sig layers
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.gc2_mu(x, adj)
        log_sig = self.gc2_sig(x, adj)
        return mu, torch.exp(log_sig)


# class GCN(nn.Module):
#     def __init__(self, nfeat, nout, dropout, adj, nhid=2):
#         super(GCN, self).__init__()
#         self.conv1 = gnn(nfeat, nout*nfeat)
#         self.conv2 = gnn(nout*nfeat, nout)
#         self.dropout = dropout
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = torch.sigmoid(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return torch.sigmoid(x)


class TGCN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""
    def __init__(self, in_feat, out_feat, G_hidden=1, seq_len=15, n_layers=2, dropout=0.1, adj=None, mode='GRU'):
        super(TGCN, self).__init__()

        # self.gcn = [GCN(in_feat, G_hidden, dropout, adj) for _ in range(seq_len)]
        self.gcn = GCNEncoder(in_feat, n_latent, G_hidden, dropout, adj)
        self.adj = adj
        self.RNN_feat = G_hidden*adj.shape[0]
        self.bn = nn.BatchNorm1d(self.RNN_feat)
        if mode == 'GRU':
            self.rnn = nn.GRU(self.RNN_feat, out_feat, n_layers, dropout=dropout)

        self.linear1 = nn.Linear(out_feat, int(out_feat/2))

    def forward(self, x):
        '''the shape of x: (seq_len, Data_batch)
        the shape of out: (seq_len, batch_size, target_feat)'''
        # output = torch.stack([self.bn(self.gcn[i](xx).reshape(-1, self.RNN_feat)) for i, xx in enumerate(x)])
        # output = torch.stack([self.bn(self.gcn(xx).reshape(-1, self.RNN_feat)) for i, xx in enumerate(x)])
        output = torch.stack([self.gcn(xx).reshape(-1, self.RNN_feat) for i, xx in enumerate(x)])
        output, hn = self.rnn(output)
        output = self.linear(output)
        return output, hn


class RNN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""
    def __init__(self, in_feat, out_feat, n_layers=2, dropout=0.1, mode='GRU'):
        super(RNN, self).__init__()
        if mode == 'GRU':
            self.rnn = nn.GRU(in_feat, out_feat, n_layers, dropout=dropout)
        self.linear = nn.Linear(out_feat, 1)

    def forward(self, x):
        '''the shape of x: (seq_len, batch_size, num_nodes, nodes_feature)
        the shape of out: (seq_len, batch_size, out_feat)'''
        out = torch.stack([xx.reshape(xx.shape[0], -1) for xx in x])
        out, hn = self.rnn(out)
        # out = torch.sigmoid(out)
        out = self.linear(out)
        return out, hn


class Loss(nn.Module):
    def __init__(self, method):
        super(Loss, self).__init__()
        self.method = method

    def forward(self, x, y):
        '''the shape of input is (batch_size, features)'''
        if self.method == 'rmse':
            return torch.norm(x-y, p=2)
        elif self.method == 'mae':
            return torch.norm(x-y, p=1)