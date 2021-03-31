from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import PackedSequence
#from lstm_xue import LSTM, BayesLSTM
#from locked_dropout import LockedDropout


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
            # getattr for baseline LSTM.
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = BayesLSTM(ninp, nhid, nlayers, position=1, dropout=dropout)
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


class BayesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, position=0, bias=True, dropout=0.):
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
            self.weight_hh_lgstd_1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.weight_ih_lgstd_1 = nn.Parameter(torch.Tensor(hidden_size, input_size))
            self.bias_hh_lgstd_1 = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_ih_lgstd_1 = nn.Parameter(torch.Tensor(hidden_size))
            pass
        pass

        if self.position == 0:
            self.weight_hh_lgstd_1 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
            self.weight_ih_lgstd_1 = nn.Parameter(torch.Tensor(gate_size, input_size))
            self.bias_hh_lgstd_1 = nn.Parameter(torch.Tensor(gate_size))
            self.bias_ih_lgstd_1 = nn.Parameter(torch.Tensor(gate_size))
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

        if 0 <= self.position <= 4:
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
        elif self.position == 0:
            weight_mean = torch.cat([self.weight_hh_mean_1, self.weight_ih_mean_1], -1)
            weight_lgstd = torch.cat([self.weight_hh_lgstd_1, self.weight_ih_lgstd_1], -1)
            bias_mean = torch.cat([self.bias_hh_mean_1, self.bias_ih_mean_1], -1)
            bias_lgstd = torch.cat([self.bias_hh_lgstd_1, self.bias_ih_lgstd_1], -1)
            pass
        else:
            weight_lgstd, weight_mean, bias_lgstd, bias_mean = 0., 0., 0., 0.
            pass
        pass

        if prior is None and 0 <= self.position <= 4:
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
            if self.position == 0:
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
    Bayesian Self-Attention. Completed on 9.20.
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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

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
        if self.use_bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_features))
            self.bias_lgstd = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0/math.sqrt(self.out_features+1)
        self.weight_mean.data.uniform_(-stdv, stdv)
        self.weight_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
        if self.use_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mean)
            bound = 1 / math.sqrt(fan_in)
            self.bias_mean = init.uniform_(self.bias_mean, -bound, bound)
            self.bias_std = init.uniform_(self.bias_lgstd, -bound, bound)

    def sample_weight_diff(self):
        if self.training:
            weight_std = torch.exp(self.weight_lgstd)
            epsilon = weight_std.new_zeros(*weight_std.size()).normal_(0, 1.5)
            weight_diff = epsilon*weight_std
            bias_diff = None
            if self.use_bias:
                bias_std = torch.exp(self.bias_lgstd)
                epsilon = bias_std.new_zeros(*bias_std.size()).normal_()
                bias_diff = epsilon*bias_std
            return weight_diff, bias_diff
        return 0.0, 0.0

    def _flat_weights(self):
        self.weight = self.weight_mean * 1.
        weight_diff, bias_diff = self.sample_weight_diff()
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
        if prior == None:
            kl = torch.mean(weight_mean**2.-weight_lgstd*2.+torch.exp(weight_lgstd*2))/2.0
            if self.use_bias:
                bias_mean = self.bias_mean
                bias_lgstd = self.bias_lgstd
                kl += torch.mean(bias_mean**2.- bias_lgstd*2.+torch.exp(bias_lgstd*2))/2.0
        return kl

    def forward(self, input):
        self._flat_weights()
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.use_bias is not None
        )


class BayesTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(BayesTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.self_attn = BayesMultiheadAttention(d_model, nhead, dropout=dropout)

        # FNN Part
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        #self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear2 = nn.Linear(dim_feedforward,d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Entering the Linear part
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(TransformerModel, self).__init__()
        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except ImportError:
        #     raise ImportError('TransformerEncoder module does not exist in '
        #                       'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformerlayers = nn.ModuleList()
        #self.transformerlayers.append(BayesTransformerEncoderLayer(ninp, nhead, nhid, dropout=0.0))
        for i in range(nlayers):
            self.transformerlayers.append(
                    TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                )
        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,
        #                                          activation)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.bayes_emb = True
        self.ninp = ninp

        """
        Bayesian word embedding. Completed on 9.23.
        Add a liner layer followed by the nn.Embedding layer, and apply Bayesian Network on it.
        """
        if self.bayes_embed is True:
            self.embed_mean = nn.Parameter(torch.Tensor(nhid, nhid), requires_grad=True)
            self.embed_lgstd = nn.Parameter(torch.Tensor(nhid, nhid), requires_grad=True)
        pass

        self.decoder = nn.Linear(ninp, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal '
                                 'to emsize.')
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
        for mod in self.transformerlayers:
            output = mod(output, src_mask=self.src_mask)

        if self.bayes_embed:
            pred_out = F.linear(output, self.embed_mean.t())
            output = self.decoder(pred_out)
        else:
            output = self.decoder(output)
        pass

        return output


class BayesTransformerModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, tie_weights=False):
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
        #self.transformerlayers.append(BayesTransformerEncoderLayer(ninp, nhead, nhid, dropout=0.0))
        #self.transformerlayers.append(BayesTransformerEncoderLayer(ninp, nhead, nhid, dropout=0.0))
        for i in range(nlayers):
            self.transformerlayers.append(
                    TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                )
        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,
        #                                          activation)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.bayes_embed = False

        """
        Bayesian word embedding. Completed on 9.23.
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

