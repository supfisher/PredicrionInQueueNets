import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution
import torch
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nout, dropout, adj, nhid=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.sigmoid(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


class TGCN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""
    def __init__(self, in_feat, out_feat, G_hidden=1, RNN_hidden=3, seq_len=15, n_layers=2, dropout=0.1, adj=None, mode='GRU'):
        super(TGCN, self).__init__()

        self.gcn = [GCN(in_feat, G_hidden, dropout, adj) for _ in range(seq_len)]
        self.adj = adj
        self.G_feat = G_hidden*adj.shape[0]
        self.RNN_feat = int(G_hidden*adj.shape[0]/3)
        self.linear = [nn.Linear(self.G_feat, self.RNN_feat) for _ in range(seq_len)]
        self.batch_norm = [nn.BatchNorm1d(self.RNN_feat) for _ in range(seq_len)]
        if mode == 'GRU':
            self.rnn = nn.GRU(self.RNN_feat, RNN_hidden, n_layers, dropout=dropout)

        self.decoder = nn.Linear(RNN_hidden, out_feat)

    def forward(self, x):
        '''the shape of x: (seq_len, Data_batch)
        the shape of out: (seq_len, batch_size, target_feat)'''
        output = torch.stack([self.batch_norm[i](
                                self.linear[i](
                                self.gcn[i](xx).reshape(-1, self.G_feat)
                                )) for i, xx in enumerate(x)])

        output, hn = self.rnn(output)
        output = self.decoder(output)
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

    def forward(self, x):
        '''the shape of x: (seq_len, batch_size, num_nodes, nodes_feature)
        the shape of out: (seq_len, batch_size, num_nodes)'''
        out = torch.stack([xx.reshape(xx.shape[0], -1) for xx in x])
        out, hn = self.rnn(out)
        return out, hn
