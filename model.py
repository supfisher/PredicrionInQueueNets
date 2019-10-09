import torch.nn as nn
import torch
import torch.nn.functional as F
from models.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, adj):
        super(GCN, self).__init__()
        self.adj = adj
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc1(x, self.adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj)
        return F.log_softmax(x, dim=1)



class TGCN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)"""
    def __init__(self, nfeat, nhid, n_layers, nout=1, dropout=0.1, adj=None, mode='GRU'):
        super(TGCN, self).__init__()
        self.gcn = GCN(nfeat, nhid, nout, dropout, adj)
        self.adj = adj
        if mode == 'GRU':
            self.rnn = nn.GRU(nout*adj.shape[0], nhid, n_layers, dropout=dropout)

    def forward(self, x):
        out = torch.stack([torch.stack([self.gcn(xx).view(-1)]) for xx in x])
        out, hn = self.rnn(out)
        return out, hn
