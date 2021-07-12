from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.nn.utils.rnn import PackedSequence


_VF = torch._C._VariableFunctions
_rnn_impls = {
    'LSTM': _VF.lstm,
    'GRU': _VF.gru,
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
}


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            #self.rnn = LSTM(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                      options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                              dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal '
                                 'to emsize.')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the
        tokens in the sequence. The positional encodings have the same dimension
        as the embeddings, so that the two can be summed. Here, we use sine and
        cosine functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5,
                 activation="relu", tie_weights=False):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except ImportError:
            raise ImportError('TransformerEncoder module does not exist in '
                              'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,
                                                 activation)
        self.transformerlayers = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal '
                                 'to emsize.')
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
                mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformerlayers(src, self.src_mask)
        output = self.decoder(output)
        return output


'''
Self build LSTM, BayesianLSTM, GPLSTM
XBY 2.20: LSTM
'''

class BayesRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False, bayes_pos=0):
        super(BayesRNNModel, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = BayesLSTM(ninp, nhid, nlayers, position=bayes_pos, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                      options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                              dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal '
                                 'to emsize.')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.dropout = float(dropout)

        # LSTM: input gate, forget gate, cell gate, output gate.
        gate_size = 4 * hidden_size

        self.weight_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.weight_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        self.bias_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size))
        self.weight_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.weight_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        self.bias_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size))
        # self.weight_ih_3 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        # self.weight_hh_3 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        # self.bias_ih_3 = nn.Parameter(torch.Tensor(gate_size))
        # # Second bias vector included for CuDNN compatibility. Only one
        # # bias vector is needed in standard definition.
        # self.bias_hh_3 = nn.Parameter(torch.Tensor(gate_size))
        self._all_weights = [k for k, v in self.__dict__.items() if '_ih' in k or '_hh' in k]

        self.reset_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih_mean_1, -stdv, stdv)
        init.uniform_(self.weight_hh_mean_1, -stdv, stdv)
        init.uniform_(self.bias_hh_mean_1, -stdv, stdv)
        init.uniform_(self.bias_ih_mean_1, -stdv, stdv)

        init.uniform_(self.weight_ih_mean_2, -stdv, stdv)
        init.uniform_(self.weight_hh_mean_2, -stdv, stdv)
        init.uniform_(self.bias_hh_mean_2, -stdv, stdv)
        init.uniform_(self.bias_ih_mean_2, -stdv, stdv)
        #
        # init.uniform_(self.weight_ih_3, -stdv, stdv)
        # init.uniform_(self.weight_hh_3, -stdv, stdv)
        # init.uniform_(self.bias_hh_3, -stdv, stdv)
        # init.uniform_(self.bias_ih_3, -stdv, stdv)

    def flat_parameters(self):
        w_hh_1 = self.weight_hh_mean_1 * 1.
        w_ih_1 = self.weight_ih_mean_1 * 1.
        b_hh_1 = self.bias_hh_mean_1 * 1.
        b_ih_1 = self.bias_ih_mean_1 * 1.

        w_hh_2 = self.weight_hh_mean_2 * 1.
        w_ih_2 = self.weight_ih_mean_2 * 1.
        b_hh_2 = self.bias_hh_mean_2 * 1.
        b_ih_2 = self.bias_ih_mean_2 * 1.

        # w_hh_3 = self.weight_hh_3 * 1.
        # w_ih_3 = self.weight_ih_3 * 1.
        # b_hh_3 = self.bias_hh_3 * 1.
        # b_ih_3 = self.bias_ih_3 * 1.

        return [w_ih_1[:, :].contiguous(), w_hh_1[:, :].contiguous(),
                b_ih_1[:].contiguous(), b_hh_1[:].contiguous(),
                w_ih_2[:, :].contiguous(), w_hh_2[:, :].contiguous(),
                b_ih_2[:].contiguous(), b_hh_2[:].contiguous()
                # w_ih_3[:, :].contiguous(), w_hh_3[:, :].contiguous(),
                # b_ih_3[:].contiguous(), b_hh_3[:].contiguous()
                ]

    @staticmethod
    def permute_hidden(hx, permutation):
        if permutation is None:
            return hx
        return hx[0].index_select(1, permutation), hx[1].index_select(1, permutation)

    def forward(self, inputs, hx=None):  # noqa: F811
        orig_input = inputs
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = inputs.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            zeros = torch.zeros(self.num_layers,
                                max_batch_size, self.hidden_size,
                                dtype=inputs.dtype, device=inputs.device)
            hx = (zeros, zeros)
            pass
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)
            pass
        pass

        # self.flatten_parameters()
        # print(self.flat_parameters()[0].size())
        if batch_sizes is None:
            result = _rnn_impls['LSTM'](inputs, hx, self.flat_parameters(), self.bias, self.num_layers,
                                        0., self.training, False, False)
            pass
        else:
            result = _rnn_impls['LSTM'](inputs, batch_sizes, hx, self.flat_parameters(), self.bias,
                                        self.num_layers, 0., self.training, False)
            pass
        pass

        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)

class BayesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, position=0, bias=True, dropout=0., bayes_pos=0):
        super(BayesLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.position = position

        # LSTM: input gate, forget gate, cell gate, output gate.
        gate_size = 4 * hidden_size

        self.weight_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.weight_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        self.bias_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size))

        self.weight_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.weight_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        self.bias_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size))

        if 1 <= self.position <= 4:
            self.weight_hh_lgstd_1 = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.weight_ih_lgstd_1 = nn.Parameter(torch.rand(hidden_size, input_size))
            self.bias_hh_lgstd_1 = nn.Parameter(torch.rand(hidden_size))
            self.bias_ih_lgstd_1 = nn.Parameter(torch.rand(hidden_size))
            pass
        pass

        if self.position == 5:
            self.weight_hh_lgstd_1 = nn.Parameter(torch.rand(gate_size, hidden_size))
            self.weight_ih_lgstd_1 = nn.Parameter(torch.rand(gate_size, input_size))
            self.bias_hh_lgstd_1 = nn.Parameter(torch.rand(gate_size))
            self.bias_ih_lgstd_1 = nn.Parameter(torch.rand(gate_size))
            pass
        pass

        self._all_weights = [k for k, v in self.__dict__.items() if '_ih' in k or '_hh' in k]
        self.reset_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih_mean_1, -stdv, stdv)
        init.uniform_(self.weight_hh_mean_1, -stdv, stdv)
        init.uniform_(self.bias_hh_mean_1, -stdv, stdv)
        init.uniform_(self.bias_ih_mean_1, -stdv, stdv)

        init.uniform_(self.weight_ih_mean_2, -stdv, stdv)
        init.uniform_(self.weight_hh_mean_2, -stdv, stdv)
        init.uniform_(self.bias_hh_mean_2, -stdv, stdv)
        init.uniform_(self.bias_ih_mean_2, -stdv, stdv)

        if 1 <= self.position <= 4:
            init.uniform_(self.weight_hh_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.weight_ih_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.bias_hh_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.bias_ih_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            pass
        elif self.position == 5:
            init.uniform_(self.weight_hh_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.weight_ih_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.bias_hh_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.bias_ih_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            pass
        pass

    def sample_weight_diff(self):
        if self.training:
            weight_hh_std = torch.exp(self.weight_hh_lgstd_1)
            epsilon = weight_hh_std.new_zeros(*weight_hh_std.size()).normal_()
            weight_hh_diff = epsilon * weight_hh_std

            weight_ih_std = torch.exp(self.weight_ih_lgstd_1)
            epsilon = weight_ih_std.new_zeros(*weight_ih_std.size()).normal_()
            weight_ih_diff = epsilon * weight_ih_std

            bias_hh_std = torch.exp(self.bias_hh_lgstd_1)
            epsilon = bias_hh_std.new_zeros(*bias_hh_std.size()).normal_()
            bias_hh_diff = epsilon * bias_hh_std

            bias_ih_std = torch.exp(self.bias_ih_lgstd_1)
            epsilon = bias_ih_std.new_zeros(*bias_ih_std.size()).normal_()
            bias_ih_diff = epsilon * bias_ih_std

            return weight_hh_diff, weight_ih_diff, bias_hh_diff, bias_ih_diff
        return 0, 0, 0, 0

    def flat_parameters(self):
        weight_hh_1 = self.weight_hh_mean_1 * 1.
        weight_ih_1 = self.weight_ih_mean_1 * 1.
        bias_hh_1 = self.bias_hh_mean_1 * 1.
        bias_ih_1 = self.bias_ih_mean_1 * 1.

        weight_hh_2 = self.weight_hh_mean_2 * 1.
        weight_ih_2 = self.weight_ih_mean_2 * 1.
        bias_hh_2 = self.bias_hh_mean_2 * 1.
        bias_ih_2 = self.bias_ih_mean_2 * 1.

        if 1 <= self.position <= 4:
            weight_hh_diff, weight_ih_diff, bias_hh_diff, bias_ih_diff = self.sample_weight_diff()
            weight_hh_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += weight_hh_diff
            weight_ih_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += weight_ih_diff
            bias_hh_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += bias_hh_diff
            bias_ih_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += bias_ih_diff
            pass
        elif self.position == 5:
            weight_hh_diff, weight_ih_diff, bias_hh_diff, bias_ih_diff = self.sample_weight_diff()
            weight_hh_2[:] += weight_hh_diff
            weight_ih_2[:] += weight_ih_diff
            bias_hh_2[:] += bias_hh_diff
            bias_ih_2[:] += bias_ih_diff
        pass

        return [weight_ih_1[:, :].contiguous(), weight_hh_1[:, :].contiguous(),
                bias_ih_1[:].contiguous(), bias_hh_1[:].contiguous(),
                weight_ih_2[:, :].contiguous(), weight_hh_2[:, :].contiguous(),
                bias_ih_2[:].contiguous(), bias_hh_2[:].contiguous()]

    def kl_divergence(self, prior=None):
        kl = 0
        if 1 <= self.position <= 4:
            weight_mean = torch.cat(
                [self.weight_hh_mean_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size],
                 self.weight_ih_mean_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size]], -1)
            weight_lgstd = torch.cat([self.weight_hh_lgstd_1, self.weight_ih_lgstd_1], -1)
            bias_mean = torch.cat(
                [self.bias_hh_mean_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size],
                 self.bias_ih_mean_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size]], -1)
            bias_lgstd = torch.cat([self.bias_hh_lgstd_1, self.bias_ih_lgstd_1], -1)
            pass
        elif self.position == 5:
            weight_mean = torch.cat([self.weight_hh_mean_1, self.weight_ih_mean_1], -1)
            weight_lgstd = torch.cat([self.weight_hh_lgstd_1, self.weight_ih_lgstd_1], -1)
            bias_mean = torch.cat([self.bias_hh_mean_1, self.bias_ih_mean_1], -1)
            bias_lgstd = torch.cat([self.bias_hh_lgstd_1, self.bias_ih_lgstd_1], -1)
            pass
        else:
            weight_lgstd, weight_mean, bias_lgstd, bias_mean = 0., 0., 0., 0.
            pass
        pass

        if prior is None and 1 <= self.position <= 5:
            kl += torch.mean(
                weight_mean ** 2. - weight_lgstd * 2. + torch.exp(weight_lgstd * 2)) / 2.  # Max uses mean in orign
            kl += torch.mean(
                bias_mean ** 2. - bias_lgstd * 2. + torch.exp(bias_lgstd * 2)) / 2.  # Max uses mean in orign
        else:
            if 1 <= self.position <= 4:
                prior = torch.cat([prior['rnns.weight_hh_mean'][
                                   (self.position - 1) * self.hidden_size:self.position * self.hidden_size],
                                   prior['rnns.weight_ih_mean'][
                                   (self.position - 1) * self.hidden_size:self.position * self.hidden_size]], -1)
            if self.position == 5:
                prior = torch.cat([prior['rnns.weight_hh_mean'], prior['weight.theta_ih_mean']], -1)
            kl += torch.sum((weight_mean - prior) ** 2. - weight_lgstd * 2. + torch.exp(weight_lgstd * 2)) / 2.
        return kl

    @staticmethod
    def permute_hidden(hx, permutation):
        if permutation is None:
            return hx
        return hx[0].index_select(1, permutation), hx[1].index_select(1, permutation)

    def forward(self, inputs, hx=None):  # noqa: F811
        orig_input = inputs
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = inputs.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            zeros = torch.zeros(self.num_layers,
                                max_batch_size, self.hidden_size,
                                dtype=inputs.dtype, device=inputs.device)
            hx = (zeros, zeros)
            pass
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)
            pass
        pass

        # self.flatten_parameters()
        # print(self.flat_parameters()[0].size())
        if batch_sizes is None:
            result = _rnn_impls['LSTM'](inputs, hx, self.flat_parameters(), self.bias, self.num_layers,
                                        0., self.training, False, False)
            pass
        else:
            result = _rnn_impls['LSTM'](inputs, batch_sizes, hx, self.flat_parameters(), self.bias,
                                        self.num_layers, 0., self.training, False)
            pass
        pass

        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


class Bayes2LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, position=0, bias=True, dropout=0., bayes_pos=0):
        super(Bayes2LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.position = position

        # LSTM: input gate, forget gate, cell gate, output gate.
        gate_size = 4 * hidden_size

        self.weight_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.weight_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        self.bias_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size))

        self.weight_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.weight_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        self.bias_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size))


        if 1 <= self.position <= 4:
            self.weight_hh_lgstd_1 = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.weight_ih_lgstd_1 = nn.Parameter(torch.rand(hidden_size, input_size))
            self.bias_hh_lgstd_1 = nn.Parameter(torch.rand(hidden_size))
            self.bias_ih_lgstd_1 = nn.Parameter(torch.rand(hidden_size))
            self.weight_hh_lgstd_2 = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.weight_ih_lgstd_2 = nn.Parameter(torch.rand(hidden_size, input_size))
            self.bias_hh_lgstd_2 = nn.Parameter(torch.rand(hidden_size))
            self.bias_ih_lgstd_2 = nn.Parameter(torch.rand(hidden_size))
            pass
        pass

        if self.position == 5:
            self.weight_hh_lgstd_1 = nn.Parameter(torch.rand(gate_size, hidden_size))
            self.weight_ih_lgstd_1 = nn.Parameter(torch.rand(gate_size, input_size))
            self.bias_hh_lgstd_1 = nn.Parameter(torch.rand(gate_size))
            self.bias_ih_lgstd_1 = nn.Parameter(torch.rand(gate_size))
            self.weight_hh_lgstd_2 = nn.Parameter(torch.rand(gate_size, hidden_size))
            self.weight_ih_lgstd_2 = nn.Parameter(torch.rand(gate_size, input_size))
            self.bias_hh_lgstd_2 = nn.Parameter(torch.rand(gate_size))
            self.bias_ih_lgstd_2 = nn.Parameter(torch.rand(gate_size))
            pass
        pass

        self._all_weights = [k for k, v in self.__dict__.items() if '_ih' in k or '_hh' in k]
        self.reset_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih_mean_1, -stdv, stdv)
        init.uniform_(self.weight_hh_mean_1, -stdv, stdv)
        init.uniform_(self.bias_hh_mean_1, -stdv, stdv)
        init.uniform_(self.bias_ih_mean_1, -stdv, stdv)

        init.uniform_(self.weight_ih_mean_2, -stdv, stdv)
        init.uniform_(self.weight_hh_mean_2, -stdv, stdv)
        init.uniform_(self.bias_hh_mean_2, -stdv, stdv)
        init.uniform_(self.bias_ih_mean_2, -stdv, stdv)

        if 1 <= self.position <= 4:
            init.uniform_(self.weight_hh_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.weight_ih_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.bias_hh_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.bias_ih_lgstd_1, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.weight_hh_lgstd_2, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.weight_ih_lgstd_2, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.bias_hh_lgstd_2, 2 * math.log(stdv), math.log(stdv))
            init.uniform_(self.bias_ih_lgstd_2, 2 * math.log(stdv), math.log(stdv))
            pass
        pass

    def sample_weight_diff(self):
        if self.training:
            weight_hh_std = torch.exp(self.weight_hh_lgstd_1)
            epsilon = weight_hh_std.new_zeros(*weight_hh_std.size()).normal_()
            weight_hh_diff = epsilon * weight_hh_std

            weight_ih_std = torch.exp(self.weight_ih_lgstd_1)
            epsilon = weight_ih_std.new_zeros(*weight_ih_std.size()).normal_()
            weight_ih_diff = epsilon * weight_ih_std

            bias_hh_std = torch.exp(self.bias_hh_lgstd_1)
            epsilon = bias_hh_std.new_zeros(*bias_hh_std.size()).normal_()
            bias_hh_diff = epsilon * bias_hh_std

            bias_ih_std = torch.exp(self.bias_ih_lgstd_1)
            epsilon = bias_ih_std.new_zeros(*bias_ih_std.size()).normal_()
            bias_ih_diff = epsilon * bias_ih_std

            weight_hh_std = torch.exp(self.weight_hh_lgstd_2)
            epsilon = weight_hh_std.new_zeros(*weight_hh_std.size()).normal_()
            weight_hh_diff_2 = epsilon * weight_hh_std

            weight_ih_std = torch.exp(self.weight_ih_lgstd_2)
            epsilon = weight_ih_std.new_zeros(*weight_ih_std.size()).normal_()
            weight_ih_diff_2 = epsilon * weight_ih_std

            bias_hh_std = torch.exp(self.bias_hh_lgstd_2)
            epsilon = bias_hh_std.new_zeros(*bias_hh_std.size()).normal_()
            bias_hh_diff_2 = epsilon * bias_hh_std

            bias_ih_std = torch.exp(self.bias_ih_lgstd_2)
            epsilon = bias_ih_std.new_zeros(*bias_ih_std.size()).normal_()
            bias_ih_diff_2 = epsilon * bias_ih_std

            return weight_hh_diff, weight_ih_diff, bias_hh_diff, bias_ih_diff, weight_hh_diff_2, weight_ih_diff_2, bias_hh_diff_2, bias_ih_diff_2
        return 0, 0, 0, 0, 0, 0, 0, 0

    def flat_parameters(self):
        weight_hh_1 = self.weight_hh_mean_1 * 1.
        weight_ih_1 = self.weight_ih_mean_1 * 1.
        bias_hh_1 = self.bias_hh_mean_1 * 1.
        bias_ih_1 = self.bias_ih_mean_1 * 1.

        weight_hh_2 = self.weight_hh_mean_2 * 1.
        weight_ih_2 = self.weight_ih_mean_2 * 1.
        bias_hh_2 = self.bias_hh_mean_2 * 1.
        bias_ih_2 = self.bias_ih_mean_2 * 1.

        if 1 <= self.position <= 4:
            weight_hh_diff, weight_ih_diff, bias_hh_diff, bias_ih_diff, weight_hh_diff_2, weight_ih_diff_2, bias_hh_diff_2, bias_ih_diff_2 = self.sample_weight_diff()
            weight_hh_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += weight_hh_diff
            weight_ih_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += weight_ih_diff
            bias_hh_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += bias_hh_diff
            bias_ih_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += bias_ih_diff
            weight_hh_2[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += weight_hh_diff_2
            weight_ih_2[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += weight_ih_diff_2
            bias_hh_2[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += bias_hh_diff_2
            bias_ih_2[(self.position - 1) * self.hidden_size:self.position * self.hidden_size] += bias_ih_diff_2
            pass
        pass

        return [weight_ih_1[:, :].contiguous(), weight_hh_1[:, :].contiguous(),
                bias_ih_1[:].contiguous(), bias_hh_1[:].contiguous(),
                weight_ih_2[:, :].contiguous(), weight_hh_2[:, :].contiguous(),
                bias_ih_2[:].contiguous(), bias_hh_2[:].contiguous()]

    def kl_divergence(self, prior=None):
        kl = 0
        if 1 <= self.position <= 4:
            weight_mean = torch.cat(
                [self.weight_hh_mean_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size],
                 self.weight_ih_mean_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size]], -1)
            weight_lgstd = torch.cat([self.weight_hh_lgstd_1, self.weight_ih_lgstd_1], -1)
            bias_mean = torch.cat(
                [self.bias_hh_mean_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size],
                 self.bias_ih_mean_1[(self.position - 1) * self.hidden_size:self.position * self.hidden_size]], -1)
            bias_lgstd = torch.cat([self.bias_hh_lgstd_1, self.bias_ih_lgstd_1], -1)
            pass
        elif self.position == 5:
            weight_mean = torch.cat([self.weight_hh_mean_1, self.weight_ih_mean_1], -1)
            weight_lgstd = torch.cat([self.weight_hh_lgstd_1, self.weight_ih_lgstd_1], -1)
            bias_mean = torch.cat([self.bias_hh_mean_1, self.bias_ih_mean_1], -1)
            bias_lgstd = torch.cat([self.bias_hh_lgstd_1, self.bias_ih_lgstd_1], -1)
            weight_mean += torch.cat([self.weight_hh_mean_2, self.weight_ih_mean_1], -1)
            weight_lgstd += torch.cat([self.weight_hh_lgstd_2, self.weight_ih_lgstd_1], -1)
            bias_mean += torch.cat([self.bias_hh_mean_2, self.bias_ih_mean_1], -1)
            bias_lgstd += torch.cat([self.bias_hh_lgstd_2, self.bias_ih_lgstd_1], -1)
            pass
        else:
            weight_lgstd, weight_mean, bias_lgstd, bias_mean = 0., 0., 0., 0.
            pass
        pass

        if prior is None and 1 <= self.position <= 5:
            kl += torch.mean(
                weight_mean ** 2. - weight_lgstd * 2. + torch.exp(weight_lgstd * 2)) / 2.  # Max uses mean in orign
            kl += torch.mean(
                bias_mean ** 2. - bias_lgstd * 2. + torch.exp(bias_lgstd * 2)) / 2.  # Max uses mean in orign
        else:
            if 1 <= self.position <= 4:
                prior = torch.cat([prior['rnns.weight_hh_mean'][
                                   (self.position - 1) * self.hidden_size:self.position * self.hidden_size],
                                   prior['rnns.weight_ih_mean'][
                                   (self.position - 1) * self.hidden_size:self.position * self.hidden_size]], -1)
            if self.position == 5:
                prior = torch.cat([prior['rnns.weight_hh_mean'], prior['weight.theta_ih_mean']], -1)
            kl += torch.sum((weight_mean - prior) ** 2. - weight_lgstd * 2. + torch.exp(weight_lgstd * 2)) / 2.
        return kl

    @staticmethod
    def permute_hidden(hx, permutation):
        if permutation is None:
            return hx
        return hx[0].index_select(1, permutation), hx[1].index_select(1, permutation)

    def forward(self, inputs, hx=None):  # noqa: F811
        orig_input = inputs
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = inputs.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            zeros = torch.zeros(self.num_layers,
                                max_batch_size, self.hidden_size,
                                dtype=inputs.dtype, device=inputs.device)
            hx = (zeros, zeros)
            pass
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)
            pass
        pass

        # self.flatten_parameters()
        # print(self.flat_parameters()[0].size())
        if batch_sizes is None:
            result = _rnn_impls['LSTM'](inputs, hx, self.flat_parameters(), self.bias, self.num_layers,
                                        0., self.training, False, False)
            pass
        else:
            result = _rnn_impls['LSTM'](inputs, batch_sizes, hx, self.flat_parameters(), self.bias,
                                        self.num_layers, 0., self.training, False)
            pass
        pass

        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


'''
Self build Transformer, BayesianTM, GPTM
XBY 2.20: Transformer
'''

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.qkv_net = nn.Linear(embed_dim, 3 * embed_dim)

        self.drop = nn.Dropout(dropout)

        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self.o_net = nn.Linear(embed_dim, embed_dim)

        # self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.qkv_net.weight)

        #if self.in_proj_bias is not None:
        nn.init.constant_(self.qkv_net.bias, 0.)
        nn.init.constant_(self.o_net.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        scaling = float(self.head_dim) ** -0.5
        tgt_len, bsz, embed_dim = query.size()

        # q, k, v size(): (seq_length, batch_size, dim_model)  e.g (100, 32, 512)
        q, k, v = self.qkv_net(query).chunk(3, dim=-1)
        q = q * scaling
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        # q, k size(): (batch_size * num_heads, seq_length, head_dim)  e.g (256, 100, 64)
        # print("q.size(), k.size(): ", q.size(), k.size())
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # attn_output_weights.size(): (batch_size * num_heads, seq_length, seq_length)  e.g (256, 100, 100)
        # print("attn_output_weights.size(): ", attn_output_weights.size())
        # if not self.training:
        #     print("attn_output_weights.size(): ", attn_output_weights.size())

        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.drop(attn_output_weights)
        #print("attn_output_weights_2.size(): ", attn_output_weights.size())

        attn_output = torch.bmm(attn_output_weights, v)
        #print("attn_output_weights_2.size(): ", attn_output_weights.size())

        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.o_net(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


class BayesMultiheadAttention(nn.Module):
    """
    Bayesian Self-Attention.
    All weights of linear transformation meet the Gaussian Distribution.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(BayesMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.q_net = nn.Linear(embed_dim, embed_dim)
        self.k_net = nn.Linear(embed_dim, embed_dim)
        self.v_net = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self.o_net = BayesLinear(embed_dim, embed_dim)

        # self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.q.weight)

        #if self.in_proj_bias is not None:
        nn.init.constant_(self.qkv_net.bias, 0.)
        nn.init.constant_(self.o_net.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        scaling = float(self.head_dim) ** -0.5
        tgt_len, bsz, embed_dim = query.size()

        q = self.q_net(query)
        k = self.k_net(key)
        v = self.v_net(value)
        q = q * scaling
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.o_net(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


class StandardTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(StandardTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None):
        # src.size(): (seq_length, batch_size, dim_model)  e.g (100, 32, 512)
        # print("src.size(): ", src.size())
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BayesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_lgstd = nn.Parameter(torch.Tensor(out_features, in_features))
        self.use_bias = bias
        self.sample = True

        if self.sample:
            self.weight_lgstd = nn.Parameter(torch.Tensor(out_features, in_features))

        if self.use_bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_features))
            if self.sample:
                self.bias_lgstd = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0/math.sqrt(self.out_features+1)
        self.weight_mean.data.uniform_(-stdv, stdv)
        if self.sample:
            self.weight_lgstd.data.uniform_(2*np.log(stdv), 1*np.log(stdv))
            if self.use_bias:
                self.bias_lgstd.data.uniform_(2*np.log(stdv), 1*np.log(stdv))

        if self.use_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mean)
            bound = 1 / math.sqrt(fan_in)
            self.bias_mean = init.uniform_(self.bias_mean, -bound, bound)
            self.bias_std = init.uniform_(self.bias_lgstd, -bound, bound)

    def sample_weight_diff(self):
        if self.training and self.sample:
            #print("sample")
            weight_std = torch.exp(self.weight_lgstd)
            epsilon = weight_std.new_zeros(*weight_std.size()).normal_(0, 1)
            weight_diff = epsilon*weight_std
            bias_diff = None
            if self.use_bias:
                bias_std = torch.exp(self.bias_lgstd)
                epsilon = bias_std.new_zeros(*bias_std.size()).normal_(0, 1)
                bias_diff = epsilon*bias_std
            return weight_diff, bias_diff
        #print("no-sample")
        return 0.0, 0.0

    def _flat_weights(self):
        self.weight = self.weight_mean * 1.
        weight_diff, bias_diff = self.sample_weight_diff()
        #weight_diff, bias_diff = 0.0, 0.0
        self.weight = self.weight + weight_diff
        if self.use_bias:
            self.bias = self.bias_mean
            self.bias = self.bias + bias_diff
        else:
            self.bias = None

    def kl_divergence(self, prior=None):
        kl = 0
        weight_mean = self.weight_mean
        weight_lgstd = self.weight_lgstd
        if self.sample:
            if prior == None:
                kl = torch.mean(weight_mean**2.-weight_lgstd*2.+torch.exp(weight_lgstd*2))/2.0
                if self.use_bias:
                    bias_mean = self.bias_mean
                    bias_lgstd = self.bias_lgstd
                    kl += torch.mean(bias_mean**2.- bias_lgstd*2.+torch.exp(bias_lgstd*2))/2.0
            else:
                prior_mean = prior['transformerlayers.0.linear2.weight_mean'].cuda()
                kl += torch.mean((weight_mean-prior_mean)**2.-weight_lgstd*2.+torch.exp(weight_lgstd*2))/2.
            pass

        return kl

    def forward(self, input):
        self._flat_weights()
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.use_bias is not None
        )


class BayesTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, bayes_pos=None):
        super(BayesTransformerEncoderLayer, self).__init__()
        self.bayes_pos = bayes_pos
        if self.bayes_pos == 'MHA':
            self.self_attn = BayesMultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        pass

        # FNN Part
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        if self.bayes_pos == 'FFN':
            self.linear2 = BayesLinear(dim_feedforward, d_model)
        else:
            self.linear2 = nn.Linear(dim_feedforward,d_model)
        pass
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Entering the Linear part
        if self.bayes_pos == 'FFN':
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        pass

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BayesTransformerModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, tie_weights=False, bayes_pos=None):
        super(BayesTransformerModel, self).__init__()
        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except ImportError:
        #     raise ImportError('TransformerEncoder module does not exist in '
        #                       'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformerlayers = nn.ModuleList()
        if bayes_pos == 'none':
            #for i in range(4):
            #    self.transformerlayers.append(BayesTransformerEncoderLayer(ninp, nhead, nhid, dropout, bayes_pos="MHA"))
            #pass
            for i in range(nlayers):
                self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
            pass
        elif bayes_pos == 'FFN':
#            self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
            self.transformerlayers.append(BayesTransformerEncoderLayer(ninp, nhead, nhid, dropout=0.2, bayes_pos=bayes_pos))
            for i in range(nlayers-1):
                self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
            pass
        elif bayes_pos == 'MHA':
            self.transformerlayers.append(BayesTransformerEncoderLayer(ninp, nhead, nhid, dropout=0.2, bayes_pos=bayes_pos))
            for i in range(nlayers-1):
                self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
            pass
        elif bayes_pos == 'EMB':
            for i in range(nlayers):
                self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
            pass
        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,
        #                                          activation)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        if bayes_pos == 'EMB':
            self.bayes_embed = True
        else:
            self.bayes_embed = False
        pass

        """
        Bayesian word embedding.
        Add a liner layer followed by the nn.Embedding layer, and apply Bayesian Network on it.
        """
        if self.bayes_embed is True:
            self.embed_mean = nn.Parameter(torch.Tensor(ninp, ninp), requires_grad=True)
            self.embed_lgstd = nn.Parameter(torch.Tensor(ninp, ninp), requires_grad=True)
        pass

        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal '
            #                     'to emsize.')
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def embed_sample_diff(self):
        if self.training:
            weight_std = torch.exp(self.embed_lgstd)
            epsilon = weight_std.new_zeros(*weight_std.size()).normal_()
            weight_diff = epsilon * weight_std
            return weight_diff
        return 0

    def embed_kl_divergence(self):
        kl = 0
        theta_mean = self.embed_mean
        theta_std = self.embed_lgstd
        kl += torch.mean(theta_mean ** 2. - theta_std * 2. + torch.exp(theta_std * 2)) / 2.
        return kl

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
                mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        if self.bayes_embed is True:
            stde = 1. / math.sqrt(self.ninp + 1)
            self.embed_mean.data.uniform_(-stde, stde)
            self.embed_lgstd.data.uniform_(2*np.log(stde), 1*np.log(stde))

    def forward(self, src, has_mask=True):
        # src.size(): (seq_length, batch_size)  e.g (100, 32)
        # print("src.size(): ", src.size())
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.ninp)

        if self.bayes_embed is True:
            embed_weight = self.embed_mean*1
            embed_diff = self.embed_sample_diff()
            embed_weight += embed_diff
            src =  F.linear(src, embed_weight)
        pass

        src = self.pos_encoder(src)
        # output = self.transformerlayers(src, self.src_mask)
        output = src

        # output.size(): (seq_length, batch_size, dim_model) e.g (100, 32, 512)
        # print("output.size(): ", output.size())
        for mod in self.transformerlayers:
            output = mod(output, src_mask=self.src_mask)

        if self.bayes_embed:
            pred_out = F.linear(output, self.embed_mean.t())
            output = self.decoder(pred_out)
        else:
            output = self.decoder(output)
        pass

        return output


'''
Self build GPTM
XBY 3.20: Gaussian Process LSTM
'''

class GaussRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False, gauss_pos='00'):
        super(GaussRNNModel, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = GPLSTM(ninp, nhid, nlayers, dropout=dropout, gpnn_type=gauss_pos)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                      options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                              dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal '
                                 'to emsize.')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


class GaussLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0., position=0):
        super(GaussLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.pos = position
        if 1 <= position <= 5:
            self.gpnn = GPNN(hidden_size, hidden_size, act_set=['sigmoid', 'tanh', 'relu'])
            pass
        elif position == 6 or position == 7:
            self.gpnn = GPNN(hidden_size, 4 * hidden_size)
            pass
        elif position ==8:
            self.gpnn = GPNN(hidden_size, hidden_size, act_set=['sigmoid', 'tanh', 'relu'], deterministic=True)
        else:
            self.gpnn = None
        pass

        print(self.gpnn)
        # LSTM: input gate, forget gate, cell gate, output gate.
        gate_size = 4 * hidden_size

        self.weight_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.weight_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        self.bias_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size))
        self.weight_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.weight_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.bias_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size))
        # Second bias vector included for CuDNN compatibility. Only one
        # bias vector is needed in standard definition.
        self.bias_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size))

        self._all_weights = [k for k, v in self.__dict__.items() if '_ih' in k or '_hh' in k]

        self.reset_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih_mean_1, -stdv, stdv)
        init.uniform_(self.weight_hh_mean_1, -stdv, stdv)
        init.uniform_(self.bias_hh_mean_1, -stdv, stdv)
        init.uniform_(self.bias_ih_mean_1, -stdv, stdv)

        init.uniform_(self.weight_ih_mean_2, -stdv, stdv)
        init.uniform_(self.weight_hh_mean_2, -stdv, stdv)
        init.uniform_(self.bias_hh_mean_2, -stdv, stdv)
        init.uniform_(self.bias_ih_mean_2, -stdv, stdv)

    def flat_parameters(self):
        weight_hh_1 = self.weight_hh_mean_1 * 1.
        weight_ih_1 = self.weight_ih_mean_1 * 1.
        bias_hh_1 = self.bias_hh_mean_1 * 1.
        bias_ih_1 = self.bias_ih_mean_1 * 1.

        weight_hh_2 = self.weight_hh_mean_2 * 1.
        weight_ih_2 = self.weight_ih_mean_2 * 1.
        bias_hh_2 = self.bias_hh_mean_2 * 1.
        bias_ih_2 = self.bias_ih_mean_2 * 1.

        return [weight_ih_1[:, :].contiguous(), weight_hh_1[:, :].contiguous(),
                bias_ih_1[:].contiguous(), bias_hh_1[:].contiguous(),
                weight_ih_2[:, :].contiguous(), weight_hh_2[:, :].contiguous(),
                bias_ih_2[:].contiguous(), bias_hh_2[:].contiguous()]

    def forward(self, inputs, hid):
        use_torch = False
        if use_torch is True:
            max_batch_size = inputs.size(1)
            if hid is None:
                zeros = torch.zeros(self.num_layers,
                                    max_batch_size, self.hidden_size,
                                    dtype=inputs.dtype, device=inputs.device)
                hid = (zeros, zeros)
                pass

            result = _rnn_impls['LSTM'](inputs, hid, self.flat_parameters(), self.bias, self.num_layers,
                                        0., self.training, False, False)
            outputs = result[0]
            hid = result[1:]
        else:
            outputs = None
            for i in range(inputs.size(0)):
                inp = inputs[i, :, :]
                hid = self.lstm(inp, hid)
                #print(hid[0].size())
                if outputs is None:
                    outputs = hid[0][1, :, :].unsqueeze(0)
                else:
                    outputs = torch.cat((outputs, hid[0][1, :, :].unsqueeze(0)), dim=0)
                pass
            pass
        pass

        return outputs, hid

    def lstm(self, inputs, hidden):
        hx, cx = hidden
        #print(hx.size())
        hx1 = hx[0, :, :]
        cx1 = cx[0, :, :]
        hx2 = hx[1, :, :]
        cx2 = cx[1, :, :]

        w_hh_1 = self.weight_hh_mean_1 * 1.
        w_ih_1 = self.weight_ih_mean_1 * 1.
        b_hh_1 = self.bias_hh_mean_1 * 1.
        b_ih_1 = self.bias_ih_mean_1 * 1.

        w_hh_2 = self.weight_hh_mean_2 * 1.
        w_ih_2 = self.weight_ih_mean_2 * 1.
        b_hh_2 = self.bias_hh_mean_2 * 1.
        b_ih_2 = self.bias_ih_mean_2 * 1.

        # GPact in different positions, only applying GPact in 1-layer LSTM
        # 1: input_gate, 2: forget_gate, 3: cell_gate, 4: output_gate, 5: cell, 6: hidden, 7: inputs
        if self.pos == 0:
            gates_1 = F.linear(inputs, w_ih_1.contiguous(), b_ih_1.contiguous()) \
                    + F.linear(hx1, w_hh_1.contiguous(), b_hh_1.contiguous())
            ingate_1, forgetgate_1, cellgate_1, outgate_1 = gates_1.chunk(4, 1)
            #print("No Gauss")
            pass
        elif self.pos == 1:
            gates_1 = F.linear(inputs, w_ih_1.contiguous(), b_ih_1.contiguous()) \
                    + F.linear(hx1, w_hh_1.contiguous(), b_hh_1.contiguous())
            _, forgetgate_1, cellgate_1, outgate_1 = gates_1.chunk(4, 1)
            ingate_1 = self.gpnn(inputs)
            pass
        elif self.pos == 2:
            gates_1 = F.linear(inputs, w_ih_1.contiguous(), b_ih_1.contiguous()) \
                    + F.linear(hx1, w_hh_1.contiguous(), b_hh_1.contiguous())
            ingate_1, _, cellgate_1, outgate_1 = gates_1.chunk(4, 1)
            forgetgate_1 = self.gpnn(inputs)
            pass
        elif self.pos == 3 or self.pos ==8:
            gates_1 = F.linear(inputs, w_ih_1.contiguous(), b_ih_1.contiguous()) \
                    + F.linear(hx1, w_hh_1.contiguous(), b_hh_1.contiguous())
            ingate_1, forgetgate_1, _, outgate_1 = gates_1.chunk(4, 1)
            #cellgate_1 = self.gpnn(inputs)
            pass
        elif self.pos == 4:
            gates_1 = F.linear(inputs, w_ih_1.contiguous(), b_ih_1.contiguous()) \
                    + F.linear(hx1, w_hh_1.contiguous(), b_hh_1.contiguous())
            ingate_1, forgetgate_1, cellgate_1, _ = gates_1.chunk(4, 1)
            outgate_1 = self.gpnn(inputs)
            pass
        elif self.pos == 5:
            gates_1 = F.linear(inputs, w_ih_1.contiguous(), b_ih_1.contiguous()) \
                    + F.linear(hx1, w_hh_1.contiguous(), b_hh_1.contiguous())
            ingate_1, forgetgate_1, cellgate_1, outgate_1 = gates_1.chunk(4, 1)
            cx1 = self.gpnn(cx1)
            pass
        elif self.pos == 6:
            gates_1 = F.linear(inputs, w_ih_1.contiguous(), b_ih_1.contiguous()) \
                      + self.gpnn(hx1)
            ingate_1, forgetgate_1, cellgate_1, outgate_1 = gates_1.chunk(4, 1)
            pass
        elif self.pos == 7:
            gates_1 = self.gpnn(inputs) \
                      + F.linear(hx1, w_hh_1.contiguous(), b_hh_1.contiguous())
            ingate_1, forgetgate_1, cellgate_1, outgate_1 = gates_1.chunk(4, 1)
            pass
        else:
            ingate_1, forgetgate_1, cellgate_1, outgate_1 = None, None, None, None
            pass
        pass

        # if self.pos == 1:
        #     ingate_1 = self.gpnn(ingate_1)
        #     pass
        # else:
        #     ingate_1 = torch.sigmoid(ingate_1)
        #     pass
        # pass
        #
        # if self.pos == 2:
        #     forgetgate_1 = self.gpnn(forgetgate_1)
        #     pass
        # else:
        #     forgetgate_1 = torch.sigmoid(forgetgate_1)
        #     pass
        # pass
        #
        # if self.pos == 3:
        #     cellgate_1 = self.gpnn(cellgate_1)
        #     pass
        # else:
        #     cellgate_1 = torch.tanh(cellgate_1)
        #     pass
        # pass
        #
        # if self.pos == 4:
        #     outgate_1 = self.gpnn(outgate_1)
        #     pass
        # else:
        #     outgate_1 = torch.sigmoid(outgate_1)
        #     pass
        # pass

        ingate_1 = torch.sigmoid(ingate_1)
        forgetgate_1 = torch.sigmoid(forgetgate_1)
        #cellgate_1 = torch.tanh(cellgate_1)
        if self.pos == 3 or self.pos ==8:
            cellgate_1 = self.gpnn(inputs)
            pass
        else:
            cellgate_1 = torch.tanh(cellgate_1)
            pass
        pass

        outgate_1 = torch.sigmoid(outgate_1)
        cell_1 = (forgetgate_1 * cx1) + (ingate_1 * cellgate_1)
        hidden_1 = outgate_1 * torch.tanh(cell_1)

        gates_2 = F.linear(hidden_1, w_ih_2.contiguous(), b_ih_2.contiguous()) \
                  + F.linear(hx2, w_hh_2.contiguous(), b_hh_2.contiguous())

        ingate_2, forgetgate_2, cellgate_2, outgate_2 = gates_2.chunk(4, 1)
        ingate_2 = torch.sigmoid(ingate_2)
        forgetgate_2 = torch.sigmoid(forgetgate_2)
        cellgate_2 = torch.tanh(cellgate_2)
        outgate_2 = torch.sigmoid(outgate_2)
        cell_2 = (forgetgate_2 * cx2) + (ingate_2 * cellgate_2)
        hidden_2 = outgate_2 * torch.tanh(cell_2)

        hidden = torch.cat((hidden_1.unsqueeze(0), hidden_2.unsqueeze(0)), dim=0)
        cell = torch.cat((cell_1.unsqueeze(0), cell_2.unsqueeze(0)), dim=0)

        return hidden, cell


class GPLSTM(nn.Module):
    '''GPact LSTM.'''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0., gpnn_type='00'):
        super(GPLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.gpnn_type = gpnn_type
        self.rnn = nn.ModuleList()
        if int(self.gpnn_type[0]) != 0:
            if len(self.gpnn_type) == 2:
                self.rnn.append(GPLSTMCell(input_size, hidden_size, gate_type=int(self.gpnn_type[0]), gpnn_type=int(self.gpnn_type[1])))
                self.rnn.append(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                        num_layers=self.num_layers-1))
            elif len(self.gpnn_type) == 3:
                self.rnn.append(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                        num_layers=self.num_layers-1))
                self.rnn.append(GPLSTMCell(input_size, hidden_size, gate_type=int(self.gpnn_type[0]), gpnn_type=int(self.gpnn_type[1])))
            else:
                self.rnn.append(GPLSTMCell(input_size, hidden_size, gate_type=int(self.gpnn_type[0]), gpnn_type=int(self.gpnn_type[1])))
                self.rnn.append(GPLSTMCell(input_size, hidden_size, gate_type=int(self.gpnn_type[2]), gpnn_type=int(self.gpnn_type[1])))
                pass
        else:
            self.rnn.append(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                    num_layers=self.num_layers, dropout=self.dropout))
        pass

    def forward(self, inputs, hidden=None):
        if int(self.gpnn_type[0]) != 0:
            if len(self.gpnn_type) == 2:
                gplstm_hids = [hidden[0][0, :, :], hidden[1][0, :, :]]
                lstm_hids = [hidden[0][1:, :, :], hidden[1][1:, :, :]]

                gplstm_oup, gplstm_hids = self.rnn[0](inputs, gplstm_hids)
                outputs, lstm_hids = self.rnn[1](gplstm_oup, lstm_hids)
                hids = torch.cat((gplstm_hids[0].unsqueeze(0), lstm_hids[0]), dim=0)
                cells = torch.cat((gplstm_hids[1].unsqueeze(0), lstm_hids[1]), dim=0)
                hiddens = (hids, cells)
            elif len(self.gpnn_type) == 3:
                lstm_hids = [hidden[0][0, :, :].unsqueeze(0), hidden[1][0, :, :].unsqueeze(0)]
                gplstm_hids = [hidden[0][1:, :, :].squeeze(0), hidden[1][1:, :, :].squeeze(0)]

                gplstm_oup, lstm_hids = self.rnn[0](inputs, lstm_hids)
                outputs, gplstm_hids = self.rnn[1](gplstm_oup, gplstm_hids)
                hids = torch.cat((lstm_hids[0], gplstm_hids[0].unsqueeze(0)), dim=0)
                cells = torch.cat((lstm_hids[1], gplstm_hids[1].unsqueeze(0)), dim=0)
                hiddens = (hids, cells)
            else:
                gplstm_hids1 = [hidden[0][0, :, :], hidden[1][0, :, :]]
                gplstm_hids2 = [hidden[0][1:, :, :].squeeze(0), hidden[1][1:, :, :].squeeze(0)]

                gplstm_oup, gplstm_hids1 = self.rnn[0](inputs, gplstm_hids1)
                outputs, gplstm_hids = self.rnn[1](gplstm_oup, gplstm_hids2)
                hids = torch.cat((gplstm_hids1[0].unsqueeze(0), gplstm_hids[0].unsqueeze(0)), dim=0)
                cells = torch.cat((gplstm_hids1[1].unsqueeze(0), gplstm_hids[1].unsqueeze(0)), dim=0)
                hiddens = (hids, cells)
                pass
        else:
            outputs, hiddens = self.rnn[0](inputs, hidden)

        return outputs, hiddens


class GPLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, gate_type=0, gpnn_type=0):
        super(GPLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_type = gate_type
        self.gpnn_type = gpnn_type

        # gpnn_type | 1-3: GPNN (new version) | 4: GPNN2 (first version).
        # gate_type: if gpnn_type == 1-3 (GPNN), there are four different positions for GPact.
        # 1: input_gate | 2: forget_gate | 3: cell_gate | 4: output_gate
        # gate_type: if gpnn_type == 4 (GPNN2), there are seven different positions for GPact. 1-4 are the same.
        # 5: cells | 6: hiddens | 7: inputs
        if self.gpnn_type <= 3:
            if self.gate_type == 3:
                self.gpnn = GPNN(self.hidden_size+self.input_size, self.hidden_size, gpnn_type=gpnn_type)
            elif self.gate_type == 1 or self.gate_type == 4:
                self.gpnn = GPNN(self.hidden_size+self.input_size, self.hidden_size, act_set=['sigmoid', 'tanh', 'relu'], gpnn_type=gpnn_type)
            elif self.gate_type == 2:
                self.gpnn = GPNN(self.hidden_size+self.input_size, self.hidden_size, act_set=['sigmoid'], gpnn_type=gpnn_type)
            elif self.gate_type == 5:
                self.gpnn = GPNN(self.input_size, self.hidden_size, gpnn_type=gpnn_type)
            elif 5 < self.gate_type <= 7:
                self.gpnn = GPNN(self.input_size, 4*self.hidden_size, gpnn_type=gpnn_type)
        elif self.gpnn_type == 4:
            if 0 < self.gate_type <= 5:
                self.gpnn = GPNN2(self.hidden_size, self.hidden_size, act_set=['sigmoid', 'relu', 'tanh'])
            elif 5 < self.gate_type <= 7:
                self.gpnn = GPNN2(self.hidden_size, self.hidden_size*4, act_set=['sigmoid', 'relu', 'tanh'])
            pass
        pass

        self.weights_ih = nn.Parameter(torch.Tensor(self.hidden_size*4, self.input_size))
        self.bias_ih = nn.Parameter(torch.Tensor(self.hidden_size*4))
        self.weights_hh = nn.Parameter(torch.Tensor(self.hidden_size*4, self.hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(self.hidden_size*4))
        self.reset_parameters()
        #self._all_weights = [k for k, v in self.__dict__.items() if '_ih' in k or '_hh' in k]

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weights_ih, -stdv, stdv)
        init.uniform_(self.weights_hh, -stdv, stdv)
        init.constant_(self.bias_ih, 0)
        init.constant_(self.bias_hh, 0)

    def forward(self, inputs, hid=None):
        if 0 < self.gate_type <= 7:
            if self.gpnn_type <= 3:
                self.gpnn.sample_parameters()

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)

        batch_size = inputs.size(1)
        outputs = None
        # Generate initial hiddens and cells
        if hid == None:
            zeros = torch.zeros(batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            hid = (zeros, zeros)
        for i in range(inputs.size(0)):
            inp = inputs[i, :, :]
            hid = self.Gplstm(inp, hid)
        # save the outputs
            if outputs is None:
                outputs = hid[0][:, :].unsqueeze(0)
            else:
                outputs = torch.cat((outputs, hid[0][:, :].unsqueeze(0)), dim=0)
        return outputs, hid

    def Gplstm(self, inp, hid):
        hx, cx = hid
        # print(inp.size(), hx.size())
        if self.gate_type == 6 and self.gpnn_type <= 4:
            gates = F.linear(inp, self.weights_ih, self.bias_ih) + self.gpnn(hx)
        elif self.gate_type == 7 and self.gpnn_type <= 4:
            gates = self.gpnn(inp) + F.linear(hx, self.weights_hh, self.bias_ih)
        else:
            gates = F.linear(inp, self.weights_ih, self.bias_ih) + F.linear(hx, self.weights_hh, self.bias_ih)
        pass

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # Very ugly.
        if self.gpnn_type <= 3:
            ingate = self.gpnn(inp, hx) if self.gate_type == 1 else torch.sigmoid(ingate)
            forgetgate = self.gpnn(inp, hx) if self.gate_type == 2 else torch.sigmoid(forgetgate)
            cellgate = self.gpnn(inp, hx) if self.gate_type == 3 else torch.tanh(cellgate)
            outgate = self.gpnn(inp, hx) if self.gate_type == 4 else torch.sigmoid(outgate)
            if self.gate_type == 5:
                cx = self.gpnn(cx)
        else:
            ingate = self.gpnn(ingate) if self.gate_type == 1 else torch.sigmoid(ingate)
            forgetgate = self.gpnn(forgetgate) if self.gate_type == 2 else torch.sigmoid(forgetgate)
            cellgate = self.gpnn(cellgate) if self.gate_type == 3 else torch.tanh(cellgate)
            outgate = self.gpnn(outgate) if self.gate_type == 4 else torch.sigmoid(outgate)
            if self.gate_type == 5:
                cx = self.gpnn(cx)
        pass

        cx = (forgetgate * cx) + (ingate * cellgate)
        hx = outgate * torch.tanh(cx)

        return hx, cx


class GPNN(nn.Module):
    '''
    0. Deterministic Weights, Deterministic Coeffs
    1. Determinsitic Weights, Bayesian Coeffs
    2. Bayesian Weights, Deterministic Coeffs
    3. Bayesian Weights, Bayesian Coeffs
    '''
    def __init__(self, input_size, output_size, act_set=['sigmoid', 'tanh', 'relu'], gpnn_type=0):
        super(GPNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gpnn_type = gpnn_type
        self.act_set = act_set

        # GPNN-0 setting
        self.weights_mean = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias_mean = nn.Parameter(torch.Tensor(output_size))
        self.coef_mean = nn.Parameter(torch.Tensor(len(act_set), output_size))
        self.softmax = nn.Softmax(dim=0)
        self.sample = False

        # Ugly
        if self.gpnn_type == 1:
            self.coef_lgstd = nn.Parameter(torch.empty(len(act_set), output_size))
        elif self.gpnn_type == 2:
            self.weights_lgstd = nn.Parameter(torch.Tensor(output_size, input_size))
            self.bias_lgstd = nn.Parameter(torch.Tensor(output_size))
        elif self.gpnn_type == 3:
            self.coef_lgstd = nn.Parameter(torch.empty(len(act_set), output_size))
            self.weights_lgstd = nn.Parameter(torch.Tensor(output_size, input_size))
            self.bias_lgstd = nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()
        self.sample_parameters()

    # Please Check this part
    def kl_divergence(self, prior=None):
        kl = 0
        if prior == None:
            if self.gpnn_type not in [0, 2]:
                # This is the KL, please check if our previous code have any bugs?
                kl += torch.mean(self.coef_mean ** 2 - self.coef_lgstd * 2. + torch.exp(self.coef_lgstd * 2) - 1) / 2.
            if self.gpnn_type not in [0, 1]:
                kl += torch.mean(
                    self.weights_mean ** 2 - self.weights_lgstd * 2. + torch.exp(self.weights_lgstd * 2) - 1) / 2.
                kl += torch.mean(self.bias_mean ** 2 - self.bias_lgstd * 2. + torch.exp(self.bias_lgstd * 2) - 1) / 2.
        return kl

    def reset_parameters(self):
        #import pdb; pdb.set_trace()
        stdv = 1. / math.sqrt(self.output_size)
        init.uniform_(self.weights_mean, -stdv, stdv)
        init.constant_(self.bias_mean, 0)  # Or can set to zero
        init.uniform_(self.coef_mean, 0, 1)
        #init.uniform_(self.coef_mean, -stdv, stdv)
        print(self.coef_mean.mean(dim=1))
        #init.constant_(self.coef_mean[1], 0)
        #init.constant_(self.coef_mean[2], 0)

        #self.weights_mean.data.uniform_(-stdv, stdv)
        #self.bias_mean.data.uniform_(-stdv, stdv)
        #self.coef_mean.data.uniform_(-stdv, stdv)

        if self.gpnn_type == 1:
            #self.coef_lgstd.data = torch.std(self.coef_mean.data, dim=1).unsqueeze(-1)
            init.uniform_(self.coef_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
        elif self.gpnn_type == 2:
            init.uniform_(self.weights_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
            init.uniform_(self.bias_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
        elif self.gpnn_type == 3:
            #self.coef_lgstd.data = torch.std(self.coef_mean.data, dim=1).unsqueeze(-1)
            init.uniform_(self.coef_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
            init.uniform_(self.weights_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
            init.uniform_(self.bias_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))

    def sample_parameters(self):
        if self.gpnn_type not in [0, 2]:
            self.coef_sample = torch.zeros(len(self.act_set), self.output_size, device=self.coef_lgstd.device).normal_()
        if self.gpnn_type not in [0, 1]:
            self.weights_sample = torch.zeros(self.output_size, self.input_size,
                                              device=self.weights_lgstd.device).normal_()
            self.bias_sample = torch.zeros(self.output_size, device=self.bias_lgstd.device).normal_()

    def forward(self, inp, hx=None):
        # XBY: hx=None for Transformer
        #import pdb; pdb.set_trace()
        self.device = next(self.parameters()).device
        if hx is not None:
            inputs = torch.cat([inp, hx], -1)
        else:
            inputs = inp

        #import pdb; pdb.set_trace()
        if self.gpnn_type in [0, 2]:
            coef = self.coef_mean
        else:
            coef = self.coef_mean + torch.exp(self.coef_lgstd) * self.coef_sample if self.training and self.sample else self.coef_mean

        if self.gpnn_type in [0, 1]:
            weights = self.weights_mean
            bias = self.bias_mean
        else:
            weights = self.weights_mean + torch.exp(self.weights_lgstd) * self.weights_sample if self.training and self.sample else self.weights_mean
            bias = self.bias_mean + torch.exp(self.bias_lgstd) * self.bias_sample if self.training and self.sample else self.bias_mean

        output = F.linear(inputs, weights, bias)
        #output = hx
        #import pdb; pdb.set_trace()
        #coef = self.softmax(coef)

        act_outputs = []
        #print(coef)
        #import pdb; pdb.set_trace()
        for i, act in enumerate(self.act_set):
#            act_outputs.append(getattr(F, act)(output))
#            coef = torch.ones_like(coef)
            act_outputs.append(getattr(F, act)(output) * coef[i])

#        output = torch.sigmoid(output)
        output = torch.sum(torch.stack(act_outputs), 0)
#        output = act_outputs[0]

        return output

    def __repr__(self):
        return self.__class__.__name__\
            + '(act_set=' + '+'.join(self.act_set)


class GPNNNode(nn.Module):
    '''
    0. Deterministic Weights, Deterministic Coeffs
    1. Determinsitic Weights, Bayesian Coeffs
    2. Bayesian Weights, Deterministic Coeffs
    3. Bayesian Weights, Bayesian Coeffs
    '''
    def __init__(self, input_size, output_size, act_set=['sigmoid', 'tanh', 'relu'], gpnn_type=0):
        super(GPNNNode, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gpnn_type = gpnn_type
        self.act_set = act_set
        self.act_num = len(self.act_set)

        # GPNN-0 setting
        self.weights_mean = nn.Parameter(torch.Tensor(self.act_num*output_size, input_size))
        self.bias_mean = nn.Parameter(torch.Tensor(self.act_num*output_size))
        self.coef_mean = nn.Parameter(torch.Tensor(self.act_num, output_size))
        self.softmax = nn.Softmax(dim=0)
        self.sample = True

        # Ugly
        if self.gpnn_type == 1:
            self.coef_lgstd = nn.Parameter(torch.Tensor(len(act_set), output_size))
        elif self.gpnn_type == 2:
            self.weights_lgstd = nn.Parameter(torch.Tensor(self.act_num*output_size, input_size))
            self.bias_lgstd = nn.Parameter(torch.Tensor(self.act_num*output_size))
        elif self.gpnn_type == 3:
            self.coef_lgstd = nn.Parameter(torch.Tensor(len(act_set), output_size))
            self.weights_lgstd = nn.Parameter(torch.Tensor(self.act_num*output_size, input_size))
            self.bias_lgstd = nn.Parameter(torch.Tensor(self.act_num*output_size))

        self.reset_parameters()
        self.sample_parameters()

    # Please Check this part
    def kl_divergence(self, prior=None):
        kl = 0
        if prior == None:
            if self.gpnn_type not in [0, 2]:
                # This is the KL, please check if our previous code have any bugs?
                kl += torch.mean(self.coef_mean ** 2 - self.coef_lgstd * 2. + torch.exp(self.coef_lgstd * 2) - 1) / 2.
            if self.gpnn_type not in [0, 1]:
                kl += torch.mean(
                    self.weights_mean ** 2 - self.weights_lgstd * 2. + torch.exp(self.weights_lgstd * 2) - 1) / 2.
                kl += torch.mean(self.bias_mean ** 2 - self.bias_lgstd * 2. + torch.exp(self.bias_lgstd * 2) - 1) / 2.
        return kl

    def reset_parameters(self):
        #import pdb; pdb.set_trace()
        stdv = 1. / math.sqrt(self.act_num*self.output_size)
        stda = 1. / math.sqrt(self.act_num)
        init.uniform_(self.weights_mean, -stdv, stdv)
        init.constant_(self.bias_mean, 0)  # Or can set to zero
        #init.uniform_(self.coef_mean, 0, 1)
        #init.uniform_(self.coef_mean, -stdv, stdv)
        init.uniform_(self.coef_mean, -stda, stda)
        print(self.coef_mean.mean(dim=1))

        if self.gpnn_type == 1:
            #self.coef_lgstd.data = torch.std(self.coef_mean.data, dim=1).unsqueeze(-1)
            init.uniform_(self.coef_lgstd, 2 * np.log(stda), 1 * np.log(stda))
        elif self.gpnn_type == 2:
            init.uniform_(self.weights_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
            init.uniform_(self.bias_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
        elif self.gpnn_type == 3:
            #self.coef_lgstd.data = torch.std(self.coef_mean.data, dim=1).unsqueeze(-1)
            init.uniform_(self.coef_lgstd, 2 * np.log(stda), 1 * np.log(stda))
            init.uniform_(self.weights_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
            init.uniform_(self.bias_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))

    def sample_parameters(self):
        if self.gpnn_type not in [0, 2]:
            self.coef_sample = torch.zeros(len(self.act_set), self.output_size, device=self.coef_lgstd.device).normal_()
        if self.gpnn_type not in [0, 1]:
            self.weights_sample = torch.zeros(self.act_num*self.output_size, self.input_size,
                                              device=self.weights_lgstd.device).normal_()
            self.bias_sample = torch.zeros(self.act_num*self.output_size, device=self.bias_lgstd.device).normal_()

    def forward(self, inp, hx=None):
        # XBY: hx=None for Transformer
        #import pdb; pdb.set_trace()
        self.device = next(self.parameters()).device
        if hx is not None:
            inputs = torch.cat([inp, hx], -1)
        else:
            inputs = inp

        #import pdb; pdb.set_trace()
        if self.gpnn_type in [0, 2]:
            coef = self.coef_mean
        else:
            coef = self.coef_mean + torch.exp(self.coef_lgstd) * self.coef_sample if self.training else self.coef_mean

        if self.gpnn_type in [0, 1]:
            weights = self.weights_mean
            bias = self.bias_mean
        else:
            weights = self.weights_mean + torch.exp(self.weights_lgstd) * self.weights_sample if self.training else self.weights_mean
            bias = self.bias_mean + torch.exp(self.bias_lgstd) * self.bias_sample if self.training else self.bias_mean

        output = F.linear(inputs, weights, bias)
        #output = hx
        #import pdb; pdb.set_trace()
        #coef = self.softmax(coef)
        #print(output.size())

        act_outputs = []
        #print(coef)
        #import pdb; pdb.set_trace()
        for i, act in enumerate(self.act_set):
#            act_outputs.append(getattr(F, act)(output))
#            coef = torch.ones_like(coef)
            act_outputs.append(getattr(F, act)(output[:, i*self.output_size:(i+1)*self.output_size]) * coef[i])

#        output = torch.sigmoid(output)
        output = torch.sum(torch.stack(act_outputs), 0)
#        output = act_outputs[0]

        return output

    def __repr__(self):
        return self.__class__.__name__\
            + '(act_set=' + '+'.join(self.act_set)


class GPNN2(nn.Module):

    """ Gaussian Process Neural Network """

    def __init__(self, input_dim, output_dim, n_MC_terms=150,
                 act_set={'sigmoid', 'tanh', 'relu', 'gelu'},
                 skip_act=True, deterministic=False, update_prior=True):
        super(GPNN2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_MC_terms = n_MC_terms
        self.act_set = act_set
        self.skip_act = skip_act
        self.deterministic = deterministic
        self.update_prior = update_prior
        self.frequency_mean = nn.Parameter(torch.Tensor(input_dim, n_MC_terms))
        self.frequency_lgstd = nn.Parameter(torch.Tensor(input_dim, n_MC_terms))
        self.coef = nn.Linear(n_MC_terms, output_dim)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.n_MC_terms)
        self.frequency_mean.data.uniform_(-stdv, stdv)
        self.frequency_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))

    def forward(self, input):
        if(self.training and not self.deterministic):
            frequency_std = torch.exp(self.frequency_lgstd)
            epsilon = frequency_std.new_zeros([self.input_dim, self.n_MC_terms]).normal_()
            frequency = self.frequency_mean+epsilon*frequency_std
        else:
            frequency = self.frequency_mean
        output = input.matmul(frequency)
        act_outputs = [output] if self.skip_act else []
        for act in self.act_set:
            if(act == 'sin' or act == 'cos'):
                act_outputs.append(getattr(torch, act)(output))
            else:
                act_outputs.append(getattr(F, act)(output))
        output = torch.sum(torch.stack(act_outputs), 0)
        return self.coef(output/math.sqrt(self.n_MC_terms))

    def kl_divergence(self):
        device = next(self.parameters()).device
        self.frequency_mean_prior = self.frequency_mean_prior.to(device)
        self.frequency_lgstd_prior = self.frequency_lgstd_prior.to(device)
        frequency_var = torch.exp(2*self.frequency_lgstd)
        frequency_var_prior = torch.exp(2*self.frequency_lgstd_prior)
        mean_square = (self.frequency_mean-self.frequency_mean_prior)**2./frequency_var_prior
        std_square = frequency_var/frequency_var_prior
        log_std_square = 2*(self.frequency_lgstd_prior-self.frequency_lgstd)/self.frequency_mean.size(1)
        return torch.sum(mean_square+std_square-log_std_square-1)/2.

    def reset_prior(self):
        self.frequency_mean_prior =\
            self.frequency_mean.new_zeros([self.input_dim, self.n_MC_terms])
        self.frequency_lgstd_prior =\
            self.frequency_lgstd.new_zeros([self.input_dim, self.n_MC_terms])
        if(self.update_prior):
            self.frequency_mean_prior.data = self.frequency_mean.data.clone()
            self.frequency_lgstd_prior.data = self.frequency_lgstd.data.clone()

    def __repr__(self):
        return self.__class__.__name__\
            + '(act_set=' + '+'.join(self.act_set)\
            + ', n_MC_terms=' + str(self.n_MC_terms)\
            + ', deterministic=' + str(self.deterministic) + ')'


## Node-level GPNN.
# class NodeGPNN(nn.Module):
#     """ Gaussian Process Neural Network """
#     def __init__(self, ninp, nhid, act_set=['sigmoid', 'tanh', 'relu'],
#                  deterministic=False):
#         super(NodeGPNN, self).__init__()
#         self.ninp = ninp
#         self.nhid = nhid
#         self.act_set = act_set
#         self.deterministic = deterministic
#         self.weight_mean = nn.Parameter(torch.empty(ninp, nhid))
#         self.bias_mean = nn.Parameter(torch.empty(self.nhid))
#         self.n_act = len(act_set)
#         self.coef_mean = nn.Parameter(torch.empty(self.n_act, self.nhid))
#         self.coef_lgstd = nn.Parameter(torch.empty(self.n_act, self.nhid))
#         self.gelu = nn.GELU()
#         self.init_parameters()
#         self.softmax = nn.Softmax(dim=0)
#
#     def init_parameters(self):
#         stdv = 1. / math.sqrt(self.nhid)
#         self.weight_mean.data.uniform_(-stdv, stdv)
#         self.coef_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
#         self.coef_mean.data.uniform_(-stdv, stdv)
#
#     def kl_divergence(self, prior=None):
#         if prior is None:
#             mean_square = (self.coef_mean-prior['rnn.gpnn.coef_mean'].to(self.device))**2.\
#             /torch.exp(prior['rnn.gpnn.coef_lgstd'].to(self.device)*2)
#             std_square = torch.exp(self.coef_lgstd.to(self.device)*2)\
#             /torch.exp(prior['rnn.gpnn.coef_lgstd'].to(self.device)*2)
#             lgvar = prior['rnn.gpnn.coef_lgstd'].to(self.device)-self.coef_lgstd.to(self.device)
#             kl_loss = 0.5*torch.sum(mean_square+std_square+2*lgvar-1)
#         else:
#             mean_square = self.coef_mean ** 2.
#             std_square = torch.exp(self.coef_lgstd.to(self.device)*2)
#             lgvar = self.coef_lgstd.to(self.device)
#             kl_loss = 0.5 * torch.sum(mean_square + std_square - 2 * lgvar)
#         return kl_loss
#
#     def forward(self, input):
#         self.device = next(self.parameters()).device
#         if self.training and not self.deterministic:
#             coef_std = torch.exp(self.coef_lgstd).to(self.device)
#             coef = self.coef_mean + torch.cuda.FloatTensor(len(self.act_set), self.nhid).normal_()*coef_std
#             # coef = self.coef_mean + coef_std.new_zeros(*coef_std.size()).normal_()
#         else:
#             coef = self.coef_mean
#         pass
#
#         coef = self.softmax(coef)
#         output = input.matmul(self.weight_mean) + self.bias_mean
#         if 'gelu' in self.act_set:
#             output = torch.relu(output)*coef[0, :]+torch.tanh(output)*coef[1, :]+torch.sigmoid(output)*coef[2, :]+self.gelu(output)*coef[3, :]
#         else:
#             output = torch.relu(output)*coef[0, :]+torch.tanh(output)*coef[1, :]+torch.sigmoid(output)*coef[2, :]
#         pass
#
#         return output
#
#     def __repr__(self):
#         return self.__class__.__name__\
#             + '(act_set=' + '+'.join(self.act_set)\
#             + ', deterministic=' + str(self.deterministic) + ')'


# # Node-level-2 GPNN.
# class Node2GPNN(nn.Module):
#     def __init__(self, input_dim, output_dim, deterministic=False, best_state={}, act_set=['sigmoid', 'relu', 'tanh']):
#         super(Node2GPNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.act_set = act_set
#         self.deterministic = deterministic
#         self.best_state = best_state
#         if self.deterministic:
#             self.coef_mean  = nn.Parameter(torch.empty(len(act_set), self.output_dim))
#         else:
#             self.coef_mean = nn.Parameter(torch.empty(len(act_set), self.output_dim))
#             self.coef_mean_prior = torch.zeros(len(act_set), self.output_dim)
#             self.coef_lgstd = nn.Parameter(torch.empty(len(act_set), self.output_dim))
#             self.coef_lgstd_prior = torch.zeros(len(act_set), self.output_dim)
#         self.frequency_weight_mean = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
#         self.bias = nn.Parameter(torch.empty(self.output_dim))
#         self.softmax = nn.Softmax(dim=0)
#         self.gelu = nn.GELU()
#
#         self.init_parameters()
#
#     def init_parameters(self):
#         stdv = 1./math.sqrt(self.output_dim)
#         if self.deterministic:
#             self.coef_mean.data.uniform_(-stdv, stdv)
#             self.frequency_weight_mean.data.uniform_(-stdv, stdv)
#             self.bias.data.fill_(0.0)
#         else:
#             self.coef_mean.data.uniform_(-stdv, stdv)
#             self.frequency_weight_mean.data.uniform_(-stdv, stdv)
#             self.coef_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
#             self.bias.data.fill_(0.0)
#             # self.frequency_weight_mean.data = self.best_state['lstm.gpnn.weight_mean']
#             # self.coef_mean.data = self.best_state['lstm.gpnn.coef_mean']
#             # self.coef_mean_prior = self.best_state['lstm.gpnn.coef_mean']
#             # self.coef_lgstd.data = torch.std(self.coef_mean.data,dim=1).unsqueeze(-1)
#             # self.coef_lgstd_prior = torch.std(self.coef_mean.data,dim=1).unsqueeze(-1)
#             # self.bias.data = self.best_state['lstm.gpnn.bias']
#
#     def kl_divergence(self):
#         mean_square = (self.coef_mean-self.coef_mean_prior.to(self.device))**2.\
#         /torch.exp(self.coef_lgstd_prior.to(self.device)*2)
#         std_square = torch.exp(self.coef_lgstd.to(self.device)*2)\
#         /torch.exp(self.coef_lgstd_prior.to(self.device)*2)
#         lgvar = self.coef_lgstd_prior.to(self.device)-self.coef_lgstd.to(self.device)
#         kl_loss = 0.5*torch.sum(mean_square+std_square+2*lgvar-1)
#         return kl_loss
#
#     def forward(self, input):
#         self.device = next(self.parameters()).device
#         if self.training and not self.deterministic:
#             coef_std = torch.exp(self.coef_lgstd).to(self.device)
#             coef = self.coef_mean + torch.cuda.FloatTensor(len(self.act_set), self.output_dim).normal_()*coef_std
#         else:
#             coef = self.coef_mean
#         coef = self.softmax(coef)
#         output = input.matmul(self.frequency_weight_mean) + self.bias
#         #print(output.size())
#         if 'gelu' in self.act_set:
#             output = torch.relu(output)*coef[0, :]+torch.tanh(output)*coef[1, :]+torch.sigmoid(output)*coef[2, :]+self.gelu(output)*coef[3, :]
#         else:
#             output = torch.relu(output)*coef[0, :]+torch.tanh(output)*coef[1, :]+torch.sigmoid(output)*coef[2, :]
#         pass
#
#         return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'input_dim=' + str(self.input_dim) \
#             + ', output_dim=' + str(self.output_dim) + ')'


'''
Self build GPTM
XBY 3.28: Gaussian Process Transformer
'''

class GaussTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, gauss_pos=None):
        super(GaussTransformerEncoderLayer, self).__init__()
        self.gauss_pos = gauss_pos
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # FNN Part
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.gpnn_type = gauss_pos

        if 0 <= gauss_pos <= 3:
            self.gpnn = GPNN(d_model, dim_feedforward, act_set=['tanh', 'sigmoid', 'relu', 'gelu'], gpnn_type=gauss_pos)
        elif gauss_pos == 4:
            self.gpnn = GPNN2(d_model, dim_feedforward, act_set=['tanh', 'sigmoid', 'relu', 'gelu'])

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Entering the Linear part
        if self.gpnn_type < 4:
            self.gpnn.sample_parameters()

        src2 = self.linear2(self.dropout(self.gpnn(src)))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class GaussTransformerModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, tie_weights=False, gauss_pos=4):
        super(GaussTransformerModel, self).__init__()
        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except ImportError:
        #     raise ImportError('TransformerEncoder module does not exist in '
        #                       'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformerlayers = nn.ModuleList()
        if gauss_pos > 4:
            for i in range(nlayers):
                self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
            pass
        else:
            #self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
            self.transformerlayers.append(GaussTransformerEncoderLayer(ninp, nhead, nhid, dropout, gauss_pos))
#            self.transformerlayers.append(GaussTransformerEncoderLayer(ninp, nhead, nhid, dropout, gauss_pos))
            for i in range(nlayers-2):
                self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
            pass
            self.transformerlayers.append(GaussTransformerEncoderLayer(ninp, nhead, nhid, dropout, gauss_pos))
        pass

        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,
        #                                          activation)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal '
            #                     'to emsize.')
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
                mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        # src.size(): (seq_length, batch_size)  e.g (100, 32)
        # print("src.size(): ", src.size())
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.ninp)

        src = self.pos_encoder(src)
        # output = self.transformerlayers(src, self.src_mask)
        output = src

        # output.size(): (seq_length, batch_size, dim_model) e.g (100, 32, 512)
        # print("output.size(): ", output.size())
        for mod in self.transformerlayers:
            output = mod(output, src_mask=self.src_mask)

        output = self.decoder(output)

        return output


