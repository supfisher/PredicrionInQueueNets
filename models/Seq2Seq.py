import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import rmse, mae, mare


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.reshape(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.reshape(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.reshape(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).reshape(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights




class MyGRU(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers=2, dropout=0.2, shift_func=None):
        super(MyGRU, self).__init__()
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.grucell = nn.GRUCell(input_size=in_dim, hidden_size=self.hid_dim)

        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hid_dim, in_dim)
        )
        self.shift_func = shift_func

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
                # d = torch.ones_like(d)*-100

            # d = self.bn(d)
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
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.2):
        super(Encoder, self).__init__()
        self.fc_in = nn.Linear(in_dim, hid_dim)
        self.encoder = MyGRU(hid_dim, hid_dim, n_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, out_dim)

    def forward(self, x, h=None, begin=None, observ=True):
        x = torch.stack([self.fc_in(o) for o in x])
        out, hid, shift = self.encoder(x, observ=observ)
        output = torch.stack([self.fc_out(o) for o in out])
        # hid = self.fc_out(hid)

        end = out[-1]
        return output, hid, end


class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.2):
        super(Decoder, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(in_dim, hid_dim)
        )
        # shift_func = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim)
        # )
        shift_func = None
        self.decoder = MyGRU(hid_dim, hid_dim, n_layers=n_layers, dropout=dropout, shift_func=shift_func)
        self.fc_out = nn.Linear(hid_dim, out_dim)

    def forward(self, x, h=None, begin=None, observ=None):
        x = torch.stack([self.fc_in(o) for o in x])
        out, hid, shift = self.decoder(x, h=h, begin=begin, observ=observ)
        output = torch.stack([self.fc_out(o) for o in out])

        shift = torch.cat((begin.unsqueeze(0), shift))

        return output, shift[0:-1], x


class Seq2Seq(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.2):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(in_dim, hid_dim, out_dim=out_dim, n_layers=n_layers, dropout=dropout)

        self.decoder = Decoder(in_dim, hid_dim, out_dim=out_dim, n_layers=n_layers, dropout=dropout)

        self.fc_out = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x, observ_data, observ_target, h=None, observ=True):
        out_encoder, hid, end = self.encoder(x)

        out_decoder, shift_out, shift_in = self.decoder(observ_data, h=hid, begin=end, observ=observ)

        output = torch.cat((out_encoder, out_decoder))

        output = torch.stack([self.fc_out(o) for o in output])

        return output, shift_out, shift_in


class RNN(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout=0.2):
        super(RNN, self).__init__()
        # self.bn = nn.BatchNorm1d(num_features=in_dim)
        self.rnn = nn.GRU(in_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Sequential(
            # nn.BatchNorm1d(num_features=hid_dim),
            nn.Linear(hid_dim, hid_dim),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x):
        # x = torch.stack([self.bn(xx) for xx in x])
        out, hid = self.rnn(x)

        out = torch.stack([self.fc_out(o) for o in out])

        return out


class Seq2SeqLoss(nn.Module):
    def __init__(self):
        super(Seq2SeqLoss, self).__init__()

    def forward(self, pred, target, shift_out, shift_in, loss):
        '''the shape of input is (batch_size, features)'''
        if loss == 'l2':
            loss_target = F.l1_loss(pred.view(-1), target.view(-1))
            # loss_shift = F.l1_loss(shift_out.view(-1), shift_in.view(-1))
        elif loss == 'l1':
            loss_target = F.mse_loss(pred.view(-1), target.view(-1))
        # loss_shift = F.mse_loss(shift_out.view(-1), shift_in.view(-1))

        return loss_target


class RNNLoss(nn.Module):
    def __init__(self):
        super(RNNLoss, self).__init__()

    def forward(self, pred, target, loss):
        '''the shape of input is (batch_size, features)'''
        if loss == 'l2':
            loss_target = F.mse_loss(pred.view(-1), target.view(-1))
        elif loss == 'l1':
            loss_target = F.l1_loss(pred.view(-1), target.view(-1))
        return loss_target