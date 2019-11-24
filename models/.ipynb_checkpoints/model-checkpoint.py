import torch.nn as nn
from torch.nn.modules.transformer import Transformer
import torch.nn.functional as F
import torch
from torch_geometric.nn.conv import GraphConv as gnn
from .layers import GraphConvolution as gcn



class VAE(nn.Module):
    """Encoder using GCN layers"""
    def __init__(self, n_feat, n_hid, n_latent, dropout, adj_shape=None):
        super(VAE, self).__init__()
        self.latent = n_latent
        self.adj_shape = adj_shape
        self.gc1 = gnn(n_feat, n_hid)
        self.mu = gnn(n_hid, n_latent)
        self.logvar = gnn(n_hid, n_latent)
        self.gc2 = gnn(n_latent, n_feat)
        self.dropout = dropout

    def encode(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def inner_product(self, z):
        z = z.reshape(-1, self.adj_shape, self.latent)
        z = z.mean(0)
        adj = torch.mm(z, z.t())
        return torch.sigmoid(adj)

    def forward(self, data):
        # First layer shared between mu/sig layers
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x.float(), edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        mu = self.mu(x, edge_index)
        logvar = self.logvar(x, edge_index)
        z = self.encode(mu, logvar)
        # out = torch.cat((z, mu, logvar, self.inner_product(z)), 1)
        return z, mu, logvar, self.inner_product(z)


# class GAE(nn.Module):
#     """Encoder using GCN layers"""
#     def __init__(self, n_feat, n_hid, n_latent, dropout, adj_shape=None):
#         super(GAE, self).__init__()
#         self.latent = n_latent
#         self.adj_shape = adj_shape
#         self.gc1 = gnn(n_feat, n_hid)
#         self.gc2 = gnn(n_hid, n_latent)
#         self.dropout = dropout


#     def inner_product(self, z):
#         z = z.reshape(-1, self.adj_shape, self.latent)
#         z = z.mean(0)
#         adj = torch.mm(z, z.t())
#         return torch.sigmoid(adj)

#     def forward(self, data):
#         # First layer shared between mu/sig layers
#         x, edge_index = data.x, data.edge_index
#         z = self.gc1(x.float(), edge_index)
#         z = F.relu(z)
#         z = self.gc2(z, edge_index)
# #         z = F.relu(z)
# #         x = F.dropout(x, self.dropout, training=self.training)
#         return z, self.inner_product(z)


# class GCN(nn.Module):
#     def __init__(self, n_feat, n_hid, n_latent, dropout):
#         super(GCN, self).__init__()
#         self.conv1 = gnn(n_feat, n_hid)
#         self.conv2 = gnn(n_hid, n_latent)
#         self.dropout = dropout

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, self.edge_index)
#         x = torch.sigmoid(x)
#         x = F.relu(x)
#         x = self.conv2(x, self.edge_index)
#         return torch.sigmoid(x)


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_latent, dropout, adj):
        super(GCN, self).__init__()
        self.conv1 = gcn(n_feat, n_hid)
        self.conv2 = gcn(n_hid, n_latent)
        self.dropout = dropout
        self.adj = adj

    def forward(self, x):
        x = self.conv1(x, self.adj)
        x = torch.sigmoid(x)
        x = F.relu(x)
        x = self.conv2(x, self.adj)
        return torch.sigmoid(x)


class LR(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR, self).__init__()
        self.sequential = nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        output = self.sequential(x)
        return output


class VAE_LR(nn.Module):
    def __init__(self, in_feat, out_feat, G_hidden=1, G_latent=1, pre_len=5, dropout=0.1, edge_index=None, reg='LR'):
        super(VAE_LR, self).__init__()
        self.adj_shape = int(max(edge_index.reshape(-1)) + 1)
        self.gcn = VAE(in_feat, G_hidden, G_latent, dropout)

        self.lr_infeat = G_latent * self.adj_shape * pre_len
        if reg == 'LR':
            self.reg = LR(self.lr_infeat, out_feat)

        self.linear = nn.Linear(out_feat, 1)

    def forward(self, x):
        G_output = torch.stack([self.gcn(xx) for i, xx in enumerate(x)])
        mu = G_output[:, :, 1].mean(0)
        logvar = G_output[:, :, 2].mean(0)
        output = G_output[:, :, 0].reshape(-1, self.lr_infeat)

        output = self.reg(output)
        output = self.linear(output)
        return output, mu, logvar


class VAE_RNN(nn.Module):
    """x is given by the shape: (seq_len, batch, input_feature)
    in_feat: the input feature size
    out_feat: the hidden feature size of GRU and also the output feature size
    G_feat: the output feature size of GNN
    n_layers: is the layer of GRU"""
    def __init__(self, in_feat, out_feat, G_hidden=1, G_latent=1, n_layers=2, tar_len=1, dropout=0.1, edge_index=None, mode='GRU'):
        super(VAE_RNN, self).__init__()

        # self.gcn = [GCN(in_feat, G_hidden, dropout, adj) for _ in range(seq_len)]
        self.adj_shape = int(max(edge_index.reshape(-1)) + 1)
        self.gcn = VAE(in_feat, G_hidden, G_latent, dropout, self.adj_shape)
        self.tar_len = tar_len
        self.G_latent = G_latent

        self.RNN_feat = 3

        self.decode = nn.Linear(self.G_latent * self.adj_shape, self.RNN_feat)

        if mode == 'GRU':
            self.rnn = nn.GRU(self.RNN_feat, out_feat, n_layers, dropout=dropout)

        self.linear = nn.Linear(out_feat, 1)

    def forward(self, x):
        '''the shape of x: (seq_len, Data_batch)
        the shape of out: (seq_len, batch_size, target_feat)'''
        G_output = [self.gcn(xx) for i, xx in enumerate(x)]

        mu = torch.stack([g[1] for g in G_output]).mean(0)
        logvar = torch.stack([g[2] for g in G_output]).mean(0)
        adj_recon = torch.stack([g[3] for g in G_output]).mean(0)  # shape = (adj_shape, adj_shape)
        output = torch.stack([g[0] for g in G_output]).reshape(len(x), -1, self.G_latent * self.adj_shape)

        output = self.decode(output)

        output, _ = self.rnn(output)
        output = self.linear(output)
        return output[-self.tar_len:], mu, logvar, adj_recon


# class GAE_RNN(nn.Module):
#     """x is given by the shape: (seq_len, batch, input_feature)
#     in_feat: the input feature size
#     out_feat: the hidden feature size of GRU and also the output feature size
#     G_feat: the output feature size of GNN
#     n_layers: is the layer of GRU"""
#     def __init__(self, in_feat, out_feat, G_hidden=1, G_latent=1, n_layers=2, tar_len=1, dropout=0.1, edge_index=None, mode='GRU'):
#         super(GAE_RNN, self).__init__()

#         # self.gcn = [GCN(in_feat, G_hidden, dropout, adj) for _ in range(seq_len)]
#         self.adj_shape = int(max(edge_index.reshape(-1)) + 1)
#         self.gcn = GAE(in_feat, G_hidden, G_latent, dropout, self.adj_shape)
#         self.tar_len = tar_len
#         self.G_latent = G_latent * self.adj_shape

#         self.RNN_feat = in_feat

#         self.decode = nn.Linear(self.G_latent, self.RNN_feat)

#         if mode == 'GRU':
#             self.rnn = nn.GRU(self.RNN_feat, out_feat, n_layers, dropout=dropout)

#         self.linear = nn.Linear(out_feat, 1)

#     def forward(self, x):
#         '''the shape of x: (seq_len, Data_batch)
#         the shape of out: (seq_len, batch_size, target_feat)'''
#         G_output = [self.gcn(xx) for i, xx in enumerate(x)]

#         adj_recon = torch.stack([g[1] for g in G_output]).mean(0)  # shape = (adj_shape, adj_shape)
#         output = torch.stack([g[0] for g in G_output]).reshape(len(x), -1, self.G_latent)

#         output = self.decode(output)

#         output, _ = self.rnn(output)

#         output = self.linear(output)

#         return output[-self.tar_len:], adj_recon

    
# class RNN(nn.Module):
#     """x is given by the shape: (seq_len, batch, input_feature)
#     in_feat: the input feature size
#     out_feat: the hidden feature size of GRU and also the output feature size
#     G_feat: the output feature size of GNN
#     n_layers: is the layer of GRU"""
#     def __init__(self, in_feat, out_feat, n_layers=2, tar_len=1, dropout=0.1, mode='GRU'):
#         super(RNN, self).__init__()
#         self.tar_len = tar_len
#         if mode == 'GRU':
#             self.rnn = nn.GRU(in_feat, out_feat, n_layers, dropout=dropout)
#         self.linear = nn.Linear(out_feat, 1)

#     def forward(self, x):
#         '''the shape of x: (seq_len, batch_size, num_nodes, nodes_feature)
#         the shape of out: (seq_len, batch_size, out_feat)'''
#         out = torch.stack([xx.reshape(xx.shape[0], -1) for xx in x])
#         out, hn = self.rnn(out)
#         # out = torch.sigmoid(out)
#         out = self.linear(out)
#         return out[-self.tar_len:]



class VAE_RNNLoss(nn.Module):
    def __init__(self):
        super(VAE_RNNLoss, self).__init__()

    def forward(self, pred, target, mu, logvar, adj_recon, adj_origin):
        '''the shape of input is (batch_size, features)'''
        loss = F.mse_loss(pred, target)
        # inputs = pred - target
        # loss = torch.norm(inputs, p=2)
        G_loss = F.binary_cross_entropy(adj_recon.reshape(-1), adj_origin.reshape(-1))
        KLD = -0.5*torch.sum(1+2*logvar-mu.pow(2)-logvar.exp())/mu.shape[0]

        # return torch.max(loss, G_loss+KLD)
        return loss

# class GAE_RNNLoss(nn.Module):
#     def __init__(self):
#         super(GAE_RNNLoss, self).__init__()

#     def forward(self, pred, target, adj_recon, adj_origin):
#         '''the shape of input is (batch_size, features)'''
# #         loss = F.mse_loss(pred, target)
#         # inputs = pred - target
#         # loss = torch.norm(inputs, p=2)
#         G_loss = F.binary_cross_entropy(adj_recon.reshape(-1), adj_origin.reshape(-1))

# #         return torch.max(loss, G_loss)
# #         return loss
#         return G_loss


# class RNNLoss(nn.Module):
#     def __init__(self):
#         super(RNNLoss, self).__init__()

#     def forward(self, pred, target):
#         '''the shape of input is (batch_size, features)'''
#         loss = F.mse_loss(pred, target)

#         return loss