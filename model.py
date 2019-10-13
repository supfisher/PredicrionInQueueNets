import torch.nn as nn
import torch
import torch.nn.functional as F
from models.layers import GraphConvolution

import torch
import torch.nn.functional as F
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
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class TGCN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""
    def __init__(self, in_feat, out_feat, G_feat=1, seq_len=15, n_layers=2, dropout=0.1, adj=None, mode='GRU'):
        super(TGCN, self).__init__()

        self.gcn = [GCN(in_feat, G_feat, dropout, adj) for _ in range(seq_len)]
        self.adj = adj
        self.rnn_in_feat = G_feat*adj.shape[0]

        if mode == 'GRU':
            self.rnn = nn.GRU(self.rnn_in_feat, out_feat, n_layers, dropout=dropout)

    def forward(self, x):
        '''the shape of x: (seq_len, Data_batch)
        the shape of out: (seq_len, batch_size, target_feat)'''
        output = torch.stack([F.tanh(
                                self.gcn[i](xx).reshape(-1, self.rnn_in_feat)
                            )
                            for i, xx in enumerate(x)])
        output, hn = self.rnn(output)
        return output, hn


# class GCN(nn.Module):
#     def __init__(self, nfeat, nout, dropout, adj, nhid=2):
#         super(GCN, self).__init__()
#         self.adj = adj
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nout)
#         self.dropout = dropout
#
#     def forward(self, x):
#         x = F.relu(self.gc1(x, self.adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, self.adj)
#         return x



# class TGCN(nn.Module):
#     """x is given by the shape: (seq_len, batch, input_feature)
#     in_feat: the input feature size
#     out_feat: the hidden feature size of GRU and also the output feature size
#     G_feat: the output feature size of GNN
#     n_layers: is the layer of GRU"""
#     def __init__(self, in_feat, out_feat, G_feat=1, seq_len=15, n_layers=2, dropout=0.1, adj=None, mode='GRU'):
#         super(TGCN, self).__init__()
#
#         self.gcn = [GCN(in_feat, G_feat, dropout, adj) for _ in range(seq_len)]
#         self.adj = adj
#         if mode == 'GRU':
#             self.rnn = nn.GRU(G_feat*adj.shape[0], out_feat, n_layers, dropout=dropout)
#
#     def forward(self, x):
#         '''the shape of x: (seq_len, batch_size, num_nodes, nodes_feature)
#         the shape of out: (seq_len, batch_size, num_nodes)'''
#         # out = torch.stack([self.gcn(xx).view(x.shape[1], -1) for xx in x])
#         out = torch.stack([self.gcn[i](xx).squeeze() for i, xx in enumerate(x)])
#         out, hn = self.rnn(out)
#         return out, hn
