import torch.nn as nn
import torch
import torch.nn.functional as F
from models.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nout, dropout, adj, nhid=2):
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
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""
    def __init__(self, in_feat, out_feat, G_feat=1, n_layers=2, dropout=0.1, adj=None, mode='GRU'):
        super(TGCN, self).__init__()
        self.gcn = GCN(in_feat, G_feat, dropout, adj)
        self.adj = adj
        if mode == 'GRU':
            self.rnn = nn.GRU(G_feat*adj.shape[0], out_feat, n_layers, dropout=dropout)

    def forward(self, x):
        '''the shape of x: (seq_len, batch_size, num_nodes, nodes_feature)'''
        out = torch.stack([self.gcn(xx).view(x.shape[1], -1) for xx in x])
        out, hn = self.rnn(out)
        return out, hn
