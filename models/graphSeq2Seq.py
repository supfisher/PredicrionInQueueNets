import torch.nn as nn
import torch.nn.functional as F
import torch
from .layers import *
from torch_geometric.nn import GCNConv as GNN


class MyGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, fc_out=None, adj=None):
        super(MyGNN, self).__init__()
        self.gnn1 = GNN(in_dim, in_dim)
        self.gnn2 = GNN(in_dim, in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.adj_shape = adj.shape[0]

        self.fc_in = nn.Linear(in_dim, out_dim)
        self.fc_out = fc_out

    def forward(self, x, adj1, observ=True):
        batch_size = x[0].num_graphs
        gnn_out = list()

        for i in range(len(x)):
            if i==1 and observ:
                d = self.gnn1(x[i].x.float(), x[i].edge_index)
                d = F.dropout(d, p=0.2)
                d = self.gnn2(d, x[i].edge_index)
                d = d.view(batch_size, self.adj_shape, -1)
                d = d[adj1[i][:, 0], adj1[i][:, 1]]
                d = self.linear(d)
            else:
                d = x[i].x.float().view(batch_size, self.adj_shape, -1)
                d = d[adj1[i][:, 0], adj1[i][:, 1]]
                d = self.fc_in(d)

            if self.fc_out is not None:
                d = self.fc_out(d)

            gnn_out.append(d)

        gnn_out = torch.stack([o for o in gnn_out])
        return gnn_out


class MyGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers=2, dropout=0.2, shift_func=None):
        super(MyGRU, self).__init__()
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.grucell = nn.GRUCell(input_size=in_dim, hidden_size=self.hid_dim)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hid_dim, in_dim)
        )
        self.shift_func = None
        

    def forward_1cell(self, x, h):
        hid = list()
        for i in range(self.n_layers-1):
            x = self.grucell(x, h[i])
            hid.append(x.clone())
            x = self.fc(x)

        x = self.grucell(x, h[self.n_layers-1])
        hid.append(x.clone())

        return x, torch.stack(hid)

    def forward(self, x, h=None, begin=None, observ=True):
        if h is None:
            h = self.init_hidden(x.shape[1])
        out = list()

        if begin is not None:
            d = begin.clone()
        else:
            d = h[0].clone()

        shift_out = list()
        for i in range(len(x)):
            if observ:
                d = x[i]
            else:
                if self.shift_func is not None:
                    d = self.shift_func(d)

            d, h = self.forward_1cell(d, h)

            out.append(d)
            if self.shift_func is not None:
                shift_out.append(self.shift_func(d))
            else:
                shift_out.append(d)

        out = torch.stack([o for o in out])
        shift_out = torch.stack([o for o in shift_out])
        return out, h, shift_out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hid_dim).zero_()
        return hidden



class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, adj, dropout=0.2):
        super(Encoder, self).__init__()
        self.gnn_fc_out = None

        self.gnn = MyGNN(in_dim, hid_dim, hid_dim, self.gnn_fc_out, adj)
        self.encoder = MyGRU(hid_dim, hid_dim, n_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x, adj1):
        shift_in = self.gnn(x, adj1)

        out, hid, shift_out = self.encoder(shift_in)

        output = torch.stack([self.fc_out(o) for o in out])

        end = out[-1]

        return output, hid, end


class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, adj, dropout=0.2):
        super(Decoder, self).__init__()
        self.gnn_fc_out = None

        self.gnn = MyGNN(in_dim, hid_dim, hid_dim, self.gnn_fc_out, adj)
        self.shift_func = None
        self.decoder = MyGRU(hid_dim, hid_dim, n_layers=n_layers, dropout=dropout, shift_func=self.shift_func)
        self.fc_out = nn.Sequential(
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x, adj1, hid, begin=None, observ=None):
        shift_in = self.gnn(x, adj1, observ)

        output, hid, shift_out = self.decoder(shift_in, hid, begin, observ)

        output = torch.stack([self.fc_out(o) for o in output])

        shift_out = torch.cat((begin.unsqueeze(0), shift_out))
        return output, shift_out[0:-1], shift_in


class GraphSeq2Seq(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, adj, dropout=0.2):
        super(GraphSeq2Seq, self).__init__()

        self.encoder = Encoder(in_dim, hid_dim, out_dim, n_layers=n_layers, adj=adj, dropout=dropout)

        self.decoder = Decoder(in_dim, hid_dim, out_dim, n_layers=n_layers, adj=adj, dropout=dropout)

        self.adj = adj

        self.adj_shape = adj.shape[0]

        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x, observ_data, adj1, adj1observ, observ=True):

        out_encoder, hid, end = self.encoder(x, adj1)

        out_decoder, shift_out, shift_in = self.decoder(observ_data, adj1observ, hid, begin=end, observ=observ)

        output = torch.cat((out_encoder, out_decoder))

        output = torch.stack([self.fc(out) for out in output])

        return output, shift_out, shift_in


class GraphSeq2SeqLoss(nn.Module):
    def __init__(self):
        super(GraphSeq2SeqLoss, self).__init__()

    def forward(self, pred, target, shift_out, shift_in, loss):
        '''the shape of input is (batch_size, features)'''
        if loss == 'l2':
            loss = F.mse_loss(pred.view(-1), target.view(-1))
        elif loss == 'l1':
            loss = F.l1_loss(pred.view(-1), target.view(-1))
        return loss
