import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch_geometric.nn import GCNConv as gcn
from .layers import GraphConvolution as gcn
import copy


class GCN(nn.Module):
    def __init__(self, n_feat, n_hidden, n_latent, adj, act=F.relu):
        super(GCN, self).__init__()
        self.adj = adj
        self.gcn1 = gcn(n_feat, n_hidden)
        self.act = act
        self.gcn2 = gcn(n_hidden, n_latent)


    def forward(self, x, q_ids):
        """x:(batch_size, adj_shape, hidden_feat_shape)
        q_id:(batch_size, 1)
        out: (batch_size, 1, n_latent)"""
        x = torch.stack([self.act(self.gcn1(xx, self.adj)) for xx in x])

        out = torch.stack([self.gcn2(xx, self.adj[q_id].unsqueeze(0)) for xx, q_id in zip(x, q_ids)]).squeeze()

        return out


class GCNRNNCell(nn.Module):
    def __init__(self, in_feat, out_feat, adj, mode='GRU', bias=True):
        super(GCNRNNCell, self).__init__()
        if mode == 'GRU':
            self.rnn = nn.GRUCell(in_feat, out_feat, bias=True)
        self.gcn = GCN(out_feat, out_feat, out_feat, adj)
        self.adj = adj

    def forward(self, x, hid, x_hid):
        """x: (batch_size, in_feat_shape)
        hid:(batch_size, out_feat_shape)
        x_hid: (batch_size, adj_shape, out_feat_shape)
        out: (batch_size, out_feat_shape)"""
        out = self.rnn(x, hid) #(batch_size, out_feat)
        q_id = x[:, -1].long() #(batch_size, 1)

        # for i in range(len(x_hid)): ##TODO
        # x_hid[:, q_id[:]] += out[:].clone()

        hid = self.gcn(x_hid, q_id)

        return out, hid, x_hid


    
class GCN_RNN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""
    def __init__(self, in_feat, out_feat, n_layers=2, tar_len=1, dropout=0.1, adj=None, mode='GRU'):
        super(GCN_RNN, self).__init__()

        self.tar_len = tar_len
        self.n_layers = n_layers
        self.dropout = dropout
        self.out_feat = out_feat
        self.adj_shape = adj.shape[0]
        self.gcnrnn = GCNRNNCell(in_feat, out_feat, adj, mode=mode)

        self.decode = nn.Linear(out_feat, 1)

    def forward(self, data, hid=None, x_hid=None):
        '''the shape of x: (seq_len, batch_size, feature_shape)
        the shape of out: (seq_len, batch_size, out_feat)'''
        if x_hid is None:
            x_hid = torch.zeros(data[0].shape[0], self.adj_shape, self.out_feat)
            hid = torch.zeros(data[0].shape[0], self.out_feat)

        output = []

        for x in data:
            out, hid, x_hid = self.gcnrnn(x, hid, x_hid)
            output.append(out)

        out = self.decode(torch.stack(output))

        return out, hid, x_hid
    
    
class GCN_RNNLoss(nn.Module):
    def __init__(self):
        super(GCN_RNNLoss, self).__init__()

    def forward(self, pred, target):
        '''the shape of input is (batch_size, features)'''
        loss = F.mse_loss(pred, target)
        
        return loss


class RNNCell(nn.Module):
    def __init__(self, in_feat, out_feat, mode='GRU', bias=True):
        super(RNNCell, self).__init__()
        if mode == 'GRU':
            self.rnn = nn.GRUCell(in_feat, out_feat)

    def forward(self, x, hid):
        """x: (batch_size, in_feat_shape)
        hid:(batch_size, out_feat_shape)
        x_hid: (batch_size, adj_shape, out_feat_shape)
        out: (batch_size, out_feat_shape)"""
        out = self.rnn(x, hid)  # (batch_size, out_feat)

        return out


class RNN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""

    def __init__(self, in_feat, out_feat, n_layers=2, tar_len=1, dropout=0.1, mode='GRU'):
        super(RNN, self).__init__()

        self.tar_len = tar_len
        self.n_layers = n_layers
        self.dropout = dropout
        self.out_feat = out_feat
        self.gcnrnn = RNNCell(in_feat, out_feat, mode=mode)
#         self.rnn = nn.GRU(in_feat, out_feat, n_layers)

#         self.decode = nn.Linear(out_feat, 1)
        self.decode = nn.Linear(out_feat, 1)

    def forward(self, data, hid=None):
        '''the shape of x: (seq_len, batch_size, feature_shape)
        the shape of out: (seq_len, batch_size, out_feat)'''
        if hid is None:
            hid = torch.zeros(data[0].shape[0], self.out_feat)
        output = []
        for d in data:
            hid = self.gcnrnn(d, hid)
            output.append(hid)
        out = torch.stack([self.decode(out) for out in output])
#         out = torch.stack([out for out in output])
        return out[-self.tar_len:]
#         out = self.decode(out.view(data[0].shape[0], -1))
#         return out


class RNNLoss(nn.Module):
    def __init__(self):
        super(RNNLoss, self).__init__()

    def forward(self, pred, target):
        '''the shape of input is (batch_size, features)'''
        loss = F.mse_loss(pred, target)

        return loss