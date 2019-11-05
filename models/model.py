import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn.conv import SAGEConv as gnn

class VAE(nn.Module):
    """Encoder using GCN layers"""

    def __init__(self, n_feat, n_hid, n_latent, dropout, edge_index):
        super(VAE, self).__init__()
        self.gc1 = gnn(n_feat, n_hid)
        self.mu = gnn(n_hid, n_latent)
        self.logvar = gnn(n_hid, n_latent)
        self.dropout = dropout
        self.edge_index = edge_index

    def encode(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, data):
        # First layer shared between mu/sig layers
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gc1(x.float(), edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        mu = self.mu(x, edge_index)
        logvar = self.logvar(x, edge_index)
        z = self.encode(mu, logvar)
        out = torch.cat((z, mu, logvar), 1)
        return out


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_latent, dropout, edge_index):
        super(GCN, self).__init__()
        self.conv1 = gnn(n_feat, n_hid)
        self.conv2 = gnn(n_hid, n_latent)
        self.dropout = dropout
        self.edge_index = edge_index

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, self.edge_index)
        x = torch.sigmoid(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, self.edge_index)
        return torch.sigmoid(x)


class TGCN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""
    def __init__(self, in_feat, out_feat, G_model=GCN, G_hidden=1, G_latent=1, n_layers=2, dropout=0.1, edge_index=None, mode='GRU'):
        super(TGCN, self).__init__()

        # self.gcn = [GCN(in_feat, G_hidden, dropout, adj) for _ in range(seq_len)]
        self.gcn = G_model(in_feat, G_hidden, G_latent, dropout, edge_index).float()
        self.edge_index = edge_index
        self.RNN_feat = G_latent*int(max(edge_index.reshape(-1))+1)
        self.bn = nn.BatchNorm1d(self.RNN_feat)
        if mode == 'GRU':
            self.rnn = nn.GRU(self.RNN_feat, out_feat, n_layers, dropout=dropout)

        self.linear = nn.Linear(out_feat, 1)

    def forward(self, x):
        '''the shape of x: (seq_len, Data_batch)
        the shape of out: (seq_len, batch_size, target_feat)'''
        # output = torch.stack([self.bn(self.gcn[i](xx).reshape(-1, self.RNN_feat)) for i, xx in enumerate(x)])
        # output = torch.stack([self.bn(self.gcn(xx).reshape(-1, self.RNN_feat)) for i, xx in enumerate(x)])

        G_output = torch.stack([self.gcn(xx) for i, xx in enumerate(x)])
        mu = G_output[:,:,1].mean(0)
        logvar = G_output[:,:,2].mean(0)
        output = G_output[:,:,0].unsqueeze(2).reshape(G_output.shape[0], -1, self.RNN_feat)
        output, hn = self.rnn(output)
        output = self.linear(output)
        return output, mu, logvar


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


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred, target, mu, logvar):
        '''the shape of input is (batch_size, features)'''
        loss = F.mse_loss(pred, target)
        KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())/mu.shape[0]

        return (loss+KLD)