import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
from torch.autograd import Variable
from collections import namedtuple
from model import BayesMultiheadAttention, PositionalEncoding, MultiheadAttention, BayesLinear, StandardTransformerEncoderLayer, GPNN
from torch.nn.utils.rnn import PackedSequence

_VF = torch._C._VariableFunctions
_rnn_impls = {
    'LSTM': _VF.lstm,
    'GRU': _VF.gru,
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
}

INITRANGE = 0.04
TEMPERATURE = 5
bayes_type = ["none", "FFN"]


def differentiable_gumble_sample(logits):
    # print(logits)
    noise = logits.new_zeros(*logits.size()).uniform_(0, 1)
    # print(noise)
    logits_with_noise = logits - torch.log(-torch.log(noise))
    return F.softmax(logits_with_noise / TEMPERATURE, dim=-1)


class BayesTransSearchEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, bayes_pos=None):
        super(BayesTransSearchEncoderLayer, self).__init__()
        self.bayes_pos = bayes_pos

        # Search 1: Bayes Attn or Standard Attn
        #self.bayes_attn = BayesMultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # FNN Part
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

        # Search 2: Bayes FFN or Standard FFN
        self.bayes_linear2 = BayesLinear(dim_feedforward, d_model)
        self.ffn_linear2 = nn.Linear(dim_feedforward,d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(0.1)

        self.activation = nn.GELU()
        self.gumble_flag = True

    def forward(self, src, src_mask=None):
        # probs = F.softmax(self.weights, dim=-1)
        # probs = self.weights.new_zeros(*self.weights.size()).uniform_(0, 1)
        probs = self.weights
        # print(probs)
        if self.gumble_flag is True:
            probs = differentiable_gumble_sample(probs)

        src2 = torch.zeros(src.size()).to(src.device)
        src2 += self.self_attn(src, src, src, attn_mask=src_mask)[0]
        #src2 += self.bayes_attn(src, src, src, attn_mask=src_mask)[0] * probs[0][1]


        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Entering the Linear part
        src2 = torch.zeros(src.size()).to(src.device)
        src2 += self.ffn_linear2(self.dropout(self.activation(self.linear1(src)))) * probs[0][0]
        src2 += self.bayes_linear2(self.dropout2(self.activation(self.linear1(src)))) * probs[0][1]

        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class BayesTransModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(BayesTransModel, self).__init__()
        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except ImportError:
        #     raise ImportError('TransformerEncoder module does not exist in '
        #                       'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformerlayers = nn.ModuleList()


        for i in range(nlayers):
            self.transformerlayers.append(BayesTransSearchEncoderLayer(ninp, nhead, nhid, dropout).cuda())
        #            self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
        pass
        #for i in range(nlayers-3):
        #    self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout).cuda())
        #pass


        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        if tie_weights:
            #            if nhid != ninp:
            #                raise ValueError('When using the tied flag, nhid must be equal '
            #                                 'to emsize.')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.nlayers = nlayers

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
        # output = self.transformerlayers(src, self.src_mask)
        output = src

        for layer, mod in enumerate(self.transformerlayers):
            output = mod(output, src_mask=self.src_mask)

        output = self.decoder(output)
        return output


class BayesTransModelSearch(BayesTransModel):
    """docstring for QuantTransModelSearch"""

    def __init__(self, *args):
        super(BayesTransModelSearch, self).__init__(*args)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = BayesTransModel(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
        weights_data = torch.randn(self.nlayers, 1, 2).mul_(1e-3)
        #weights_data[:, :, 1] *= -1
        weights_data = weights_data.new_zeros(*weights_data.size())
        #weights_data[0, :, :] = torch.tensor([0.2, 0.8])
        self.weights = Variable(weights_data, requires_grad=True)
        print(F.softmax(self.weights, dim=-1))
        self._arch_parameters = [self.weights]
        for i, translayer in enumerate(self.transformerlayers):
            translayer.weights = self.weights[i]

    def arch_parameters(self):
        return self._arch_parameters

    # def genotype(self):
    #
    #     def _parse(probs):
    #         gene = []
    #         for i in range(self.nlayers):
    #             W = probs[i].copy()  # 3 * 4
    #             j = W.argmax(dim=-1)
    #
    #             gene.append([bayes_type[i] for i in j])
    #         return gene
    #
    #     gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
    #     genotype = Genotype(transformer=gene)
    #     return genotype


class GaussTransSearchEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, gauss_pos=3):
        super(GaussTransSearchEncoderLayer, self).__init__()
        self.gauss_pos = gauss_pos
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # FNN Part
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.gpnn_type = gauss_pos

        if 0 <= gauss_pos <= 3:
            self.gpnn = GPNN(d_model, dim_feedforward, act_set=['tanh', 'sigmoid', 'relu', 'gelu'], gpnn_type=gauss_pos).cuda()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.gumble_flag = False

    def forward(self, src, src_mask=None):
        probs = F.softmax(self.weights, dim=-1)
        #print(probs)
        if self.gumble_flag is True:
            probs = differentiable_gumble_sample(probs)
            # print("gumble")

        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Entering the Linear part
        if self.gpnn_type == 3:
            self.gpnn.sample_parameters()

        src1 = self.activation(self.linear1(src))*probs[0][0]
        src1 += self.gpnn(src)*probs[0][1]
        src2 = self.linear2(self.dropout(src1))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class GaussTransModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(GaussTransModel, self).__init__()
        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except ImportError:
        #     raise ImportError('TransformerEncoder module does not exist in '
        #                       'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformerlayers = nn.ModuleList()

        for i in range(nlayers):
            self.transformerlayers.append(GaussTransSearchEncoderLayer(ninp, nhead, nhid, dropout).cuda())
        #            self.transformerlayers.append(StandardTransformerEncoderLayer(ninp, nhead, nhid, dropout))
        pass

        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        if tie_weights:
            #            if nhid != ninp:
            #                raise ValueError('When using the tied flag, nhid must be equal '
            #                                 'to emsize.')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.nlayers = nlayers

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
        # output = self.transformerlayers(src, self.src_mask)
        output = src

        for layer, mod in enumerate(self.transformerlayers):
            output = mod(output, src_mask=self.src_mask)

        output = self.decoder(output)
        return output


class GaussTransModelSearch(GaussTransModel):
    """docstring for QuantTransModelSearch"""

    def __init__(self, *args):
        super(GaussTransModelSearch, self).__init__(*args)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = BayesTransModel(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
        weights_data = torch.randn(self.nlayers, 1, 2).mul_(1e-3)
        weights_data = weights_data.new_zeros(*weights_data.size())
        # weights_data[0, :, :] = torch.tensor([0.8, 0.2])
        self.weights = Variable(weights_data, requires_grad=True)
        print(self.weights)
        self._arch_parameters = [self.weights]
        for i, translayer in enumerate(self.transformerlayers):
            translayer.weights = self.weights[i]

    def arch_parameters(self):
        return self._arch_parameters



class GaussLSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False, gpnn_type=3):
        super(GaussLSTMModel, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = GPLSTMSearch(ninp, nhid, nlayers, dropout=dropout, gpnn_type=gpnn_type)
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


class GaussLSTMModelSearch(GaussLSTMModel):
    """docstring for ModelSearch"""

    def __init__(self, *args):
        super(GaussLSTMModelSearch, self).__init__(*args)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = GaussLSTMModel(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
        weights_data = torch.randn(self.nlayers, 3, 2).mul_(1e-3)
        weights_data = weights_data.new_zeros(*weights_data.size())
        self.weights = Variable(weights_data, requires_grad=True)
        self._arch_parameters = [self.weights]
        print(self.weights)
        self.rnn.rnn[0].weights = self.weights[0]
        self.rnn.rnn[1].weights = self.weights[1]

    def arch_parameters(self):
        return self._arch_parameters


class GPLSTMSearch(nn.Module):
    '''GPact LSTM.'''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0., gpnn_type=0):
        super(GPLSTMSearch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.gpnn_type = gpnn_type
        self.rnn = nn.ModuleList()
        self.rnn.append(GPLSTMSearchCell(input_size, hidden_size, self.gpnn_type))
        #self.rnn.append(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.num_layers-1))
        self.rnn.append(GPLSTMSearchCell(input_size, hidden_size, self.gpnn_type))

    def forward(self, inputs, hidden=None):
        gplstm_hids = [hidden[0][0, :, :], hidden[1][0, :, :]]
        lstm_hids = [hidden[0][1, :, :], hidden[1][1, :, :]]

        gplstm_oup, gplstm_hids = self.rnn[0](inputs, gplstm_hids)
        outputs, lstm_hids = self.rnn[1](gplstm_oup, lstm_hids)
        # print(type(gplstm_hids[0].unsqueeze(0)))
        hids = torch.cat((gplstm_hids[0].unsqueeze(0), lstm_hids[0].unsqueeze(0)), dim=0)
        cells = torch.cat((gplstm_hids[1].unsqueeze(0), lstm_hids[1].unsqueeze(0)), dim=0)
        hiddens = (hids, cells)

        return outputs, hiddens


class GPLSTMSearchCell(nn.Module):
    def __init__(self, input_size, hidden_size, gpnn_type=2):
        super(GPLSTMSearchCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gpnn_type = gpnn_type

        # gpnn_type | 1-3: GPNN (new version) | 4: GPNN2 (first version).
        # gate_type: if gpnn_type == 1-3 (GPNN), there are four different positions for GPact.
        # 1: input_gate | 2: forget_gate | 3: cell_gate | 4: output_gate
        # gate_type: if gpnn_type == 4 (GPNN2), there are seven different positions for GPact. 1-4 are the same.
        # 5: cells | 6: hiddens | 7: inputs
        if self.gpnn_type <= 3:
            self.gpnn_ingate = GPNN(self.hidden_size + self.input_size, self.hidden_size, act_set=['sigmoid'], gpnn_type=gpnn_type)
            self.gpnn_forgate = GPNN(self.hidden_size + self.input_size, self.hidden_size, act_set=['sigmoid'], gpnn_type=gpnn_type)
            self.gpnn_cellgate = GPNN(self.hidden_size + self.input_size, self.hidden_size, act_set=['tanh'], gpnn_type=gpnn_type)
            self.gpnn_outgate = GPNN(self.hidden_size + self.input_size, self.hidden_size, act_set=['sigmoid'], gpnn_type=gpnn_type)
            #self.gpnn_hiddens = GPNN(self.input_size, 4*self.hidden_size, gpnn_type=gpnn_type)
            #self.gpnn_inputs = GPNN(self.input_size, 4*self.hidden_size, gpnn_type=gpnn_type)

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
        if self.gpnn_type <= 3:
            self.gpnn_ingate.sample_parameters()
            self.gpnn_forgate.sample_parameters()
            self.gpnn_cellgate.sample_parameters()
            self.gpnn_outgate.sample_parameters()
            #self.gpnn_hiddens.sample_parameters()

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
        probs = F.softmax(self.weights, dim=-1)
        hx, cx = hid
        # print(inp.size(), hx.size())
        gates = F.linear(inp, self.weights_ih, self.bias_ih)
        gates += F.linear(hx, self.weights_hh, self.bias_hh)
        #gates += self.gpnn_hiddens(hx)*probs[2][1]

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)

        forgetgate = torch.sigmoid(forgetgate)

        cellgate = torch.tanh(cellgate)*probs[0][0]
        cellgate += self.gpnn_cellgate(inp, hx)*probs[0][1]

        outgate = torch.sigmoid(outgate)*probs[1][0]
        outgate += self.gpnn_outgate(inp, hx)*probs[1][1]

        cx = (forgetgate * cx) + (ingate * cellgate)
        hx = outgate * torch.tanh(cx)

        return hx, cx


class BayesLSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False):
        super(BayesLSTMModel, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = BayesLSTMSearch(ninp, nhid, nlayers, dropout=dropout)
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


class BayesLSTMModelSearch(BayesLSTMModel):
    """docstring for ModelSearch"""

    def __init__(self, *args):
        super(BayesLSTMModelSearch, self).__init__(*args)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = BayesLSTMModel(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
        weights_data = torch.randn(self.nlayers, 4, 2).mul_(1e-3)
        #weights_data = weights_data.new_zeros(*weights_data.size())
        self.weights = Variable(weights_data, requires_grad=True)
        self._arch_parameters = [self.weights]
        self.rnn.rnn[0].weights = self.weights[0]
        self.rnn.rnn[1].weights = self.weights[1]

    def arch_parameters(self):
        return self._arch_parameters


class BayesLSTMSearch(nn.Module):
    '''GPact LSTM.'''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.):
        super(BayesLSTMSearch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.rnn = nn.ModuleList()
        self.rnn.append(BayesLSTMSearchCell(input_size, hidden_size))
        self.rnn.append(BayesLSTMSearchCell(input_size, hidden_size))

    def forward(self, inputs, hidden=None):
        gplstm_hids = [hidden[0][0, :, :], hidden[1][0, :, :]]
        lstm_hids = [hidden[0][1, :, :], hidden[1][1, :, :]]

        gplstm_oup, gplstm_hids = self.rnn[0](inputs, gplstm_hids)
        outputs, lstm_hids = self.rnn[1](gplstm_oup, lstm_hids)
        # print(type(gplstm_hids[0].unsqueeze(0)))
        hids = torch.cat((gplstm_hids[0].unsqueeze(0), lstm_hids[0].unsqueeze(0)), dim=0)
        cells = torch.cat((gplstm_hids[1].unsqueeze(0), lstm_hids[1].unsqueeze(0)), dim=0)
        hiddens = (hids, cells)

        return outputs, hiddens

class BayesLSTMSearchCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BayesLSTMSearchCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bayes_ingate = Bayes(self.hidden_size + self.input_size, self.hidden_size)
        self.bayes_forgate = Bayes(self.hidden_size + self.input_size, self.hidden_size)
        self.bayes_cellgate = Bayes(self.hidden_size + self.input_size, self.hidden_size)
        self.bayes_outgate = Bayes(self.hidden_size + self.input_size, self.hidden_size)

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
        self.bayes_cellgate.sample_parameters()
        self.bayes_outgate.sample_parameters()
        self.bayes_ingate.sample_parameters()
        self.bayes_forgate.sample_parameters()

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
            hid = self.bayeslstm(inp, hid)
        # save the outputs
            if outputs is None:
                outputs = hid[0][:, :].unsqueeze(0)
            else:
                outputs = torch.cat((outputs, hid[0][:, :].unsqueeze(0)), dim=0)
        return outputs, hid

    def bayeslstm(self, inp, hid):
        probs = F.softmax(self.weights, dim=-1)
        hx, cx = hid
        # print(inp.size(), hx.size())
        gates = F.linear(inp, self.weights_ih, self.bias_ih)
        gates += F.linear(hx, self.weights_hh, self.bias_ih)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)*probs[0][0]
        ingate += torch.sigmoid(self.bayes_ingate(inp, hx))*probs[0][1]

        forgetgate = torch.sigmoid(forgetgate)*probs[1][0]
        forgetgate += torch.sigmoid(self.bayes_forgate(inp, hx))*probs[1][1]

        cellgate = torch.tanh(cellgate)*probs[2][0]
        cellgate += torch.tanh(self.bayes_cellgate(inp, hx))*probs[2][1]

        outgate = torch.sigmoid(outgate)*probs[3][0]
        outgate += torch.sigmoid(self.bayes_outgate(inp, hx))*probs[3][1]

        cx = (forgetgate * cx) + (ingate * cellgate)
        hx = outgate * torch.tanh(cx)

        return hx, cx


class BayesLSTMSearchCell1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BayesLSTMSearchCell1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bayes_ingate = Bayes(self.hidden_size + self.input_size, self.hidden_size)
        self.bayes_forgate = Bayes(self.hidden_size + self.input_size, self.hidden_size)
        self.bayes_cellgate = Bayes(self.hidden_size + self.input_size, self.hidden_size)
        self.bayes_outgate = Bayes(self.hidden_size + self.input_size, self.hidden_size)

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
        self.bayes_cellgate.sample_parameters()
        self.bayes_outgate.sample_parameters()
        self.bayes_ingate.sample_parameters()
        self.bayes_forgate.sample_parameters()

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
            hid = self.bayeslstm(inp, hid)
        # save the outputs
            if outputs is None:
                outputs = hid[0][:, :].unsqueeze(0)
            else:
                outputs = torch.cat((outputs, hid[0][:, :].unsqueeze(0)), dim=0)
        return outputs, hid

    def bayeslstm(self, inp, hid):
        probs = F.softmax(self.weights, dim=-1)
        hx, cx = hid
        # print(inp.size(), hx.size())
        gates = F.linear(inp, self.weights_ih, self.bias_ih)
        gates += F.linear(hx, self.weights_hh, self.bias_ih)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)*probs[0][1]
        ingate += torch.sigmoid(self.bayes_ingate(inp, hx))*probs[0][0]

        forgetgate = torch.sigmoid(forgetgate)*probs[1][1]
        forgetgate += torch.sigmoid(self.bayes_forgate(inp, hx))*probs[1][0]

        cellgate = torch.tanh(cellgate)*probs[2][1]
        cellgate += torch.tanh(self.bayes_cellgate(inp, hx))*probs[2][0]

        outgate = torch.sigmoid(outgate)*probs[3][1]
        outgate += torch.sigmoid(self.bayes_outgate(inp, hx))*probs[3][0]

        cx = (forgetgate * cx) + (ingate * cellgate)
        hx = outgate * torch.tanh(cx)

        return hx, cx


class Bayes(nn.Module):
    '''
    0. Deterministic Weights, Deterministic Coeffs
    1. Determinsitic Weights, Bayesian Coeffs
    2. Bayesian Weights, Deterministic Coeffs
    3. Bayesian Weights, Bayesian Coeffs
    '''
    def __init__(self, input_size, output_size):
        super(Bayes, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # GPNN-0 setting
        self.weights_mean = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias_mean = nn.Parameter(torch.Tensor(output_size))
        self.sample = False

        # Ugly
        self.weights_lgstd = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias_lgstd = nn.Parameter(torch.Tensor(output_size))

        self.reset_parameters()
        self.sample_parameters()

    # Please Check this part
    def kl_divergence(self):
        kl = 0
        if self.sample:
            kl += torch.mean(
                self.weights_mean ** 2 - self.weights_lgstd * 2. + torch.exp(self.weights_lgstd * 2) - 1) / 2.
            kl += torch.mean(self.bias_mean ** 2 - self.bias_lgstd * 2. + torch.exp(self.bias_lgstd * 2) - 1) / 2.
        return kl

    def reset_parameters(self):
        #import pdb; pdb.set_trace()
        stdv = 1. / math.sqrt(self.output_size)
        init.uniform_(self.weights_mean, -stdv, stdv)
        init.constant_(self.bias_mean, 0)  # Or can set to zero

        init.uniform_(self.weights_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))
        init.uniform_(self.bias_lgstd, 2 * np.log(stdv), 1 * np.log(stdv))

    def sample_parameters(self):
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

        weights = self.weights_mean + torch.exp(self.weights_lgstd) * self.weights_sample if self.training and self.sample else self.weights_mean
        bias = self.bias_mean + torch.exp(self.bias_lgstd) * self.bias_sample if self.training and self.sample else self.bias_mean

        output = F.linear(inputs, weights, bias)
        #output = hx
        #import pdb; pdb.set_trace()

        return output


#class BayesRNNModel(nn.Module):
#    """Container module with an encoder, a recurrent module, and a decoder."""
#    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
#                 tie_weights=False):
#        super(BayesRNNModel, self).__init__()
#
#        self.rnn_type = rnn_type
#        self.nhid = nhid
#        self.nlayers = nlayers
#        self.drop = nn.Dropout(dropout)
#        self.encoder = nn.Embedding(ntoken, ninp)
#        if rnn_type in ['LSTM', 'GRU']:
#            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
#            self.rnn = BayesLSTMSearch(ninp, nhid, nlayers, dropout=dropout)
#        else:
#            try:
#                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
#            except KeyError:
#                raise ValueError("""An invalid option for `--model` was supplied,
#                      options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
#            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
#                              dropout=dropout)
#        self.decoder = nn.Linear(nhid, ntoken)
#
#        if tie_weights:
#            if nhid != ninp:
#                raise ValueError('When using the tied flag, nhid must be equal '
#                                 'to emsize.')
#            self.decoder.weight = self.encoder.weight
#
#        self.init_weights()
#
#    def init_weights(self):
#        initrange = 0.1
#        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#        nn.init.zeros_(self.decoder.bias)
#        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
#
#    def forward(self, x, hidden):
#        emb = self.drop(self.encoder(x))
#        output, hidden = self.rnn(emb, hidden)
#        output = self.drop(output)
#        decoded = self.decoder(output)
#        return decoded, hidden
#
#    def init_hidden(self, bsz):
#        weight = next(self.parameters())
#        if self.rnn_type == 'LSTM':
#            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
#                    weight.new_zeros(self.nlayers, bsz, self.nhid))
#        return weight.new_zeros(self.nlayers, bsz, self.nhid)
#
#
#class BayesLSTMModelSearch(BayesRNNModel):
#    """docstring for ModelSearch"""
#
#    def __init__(self, *args):
#        super(BayesLSTMModelSearch, self).__init__(*args)
#        self._args = args
#        self._initialize_arch_parameters()
#
#    def new(self):
#        model_new = BayesRNNModel(*self._args)
#        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
#            x.data.copy_(y.data)
#        return model_new
#
#    def _initialize_arch_parameters(self):
#        weights_data = torch.randn(self.nlayers, 4, 2).mul_(1e-3)
#        self.weights = Variable(weights_data, requires_grad=True)
#        self._arch_parameters = [self.weights]
#        self.rnn.weights = self.weights
#
#    def arch_parameters(self):
#        return self._arch_parameters
#
#
#class BayesLSTMSearch(nn.Module):
#    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.):
#        super(BayesLSTMSearch, self).__init__()
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.bias = bias
#        self.num_layers = num_layers
#        self.dropout = float(dropout)
#        self.sample = False
#
#        # LSTM: input gate, forget gate, cell gate, output gate.
#        gate_size = 4 * hidden_size
#
#        self.weight_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size, input_size))
#        self.weight_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
#        self.bias_ih_mean_1 = nn.Parameter(torch.Tensor(gate_size))
#        # Second bias vector included for CuDNN compatibility. Only one
#        # bias vector is needed in standard definition.
#        self.bias_hh_mean_1 = nn.Parameter(torch.Tensor(gate_size))
#
#        self.weight_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size, input_size))
#        self.weight_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size, hidden_size))
#        self.bias_ih_mean_2 = nn.Parameter(torch.Tensor(gate_size))
#        # Second bias vector included for CuDNN compatibility. Only one
#        # bias vector is needed in standard definition.
#        self.bias_hh_mean_2 = nn.Parameter(torch.Tensor(gate_size))
#
#        self.weight_hh_lgstd_1 = nn.Parameter(torch.rand(gate_size, hidden_size))
#        self.weight_ih_lgstd_1 = nn.Parameter(torch.rand(gate_size, input_size))
#        self.bias_hh_lgstd_1 = nn.Parameter(torch.rand(gate_size))
#        self.bias_ih_lgstd_1 = nn.Parameter(torch.rand(gate_size))
#        self.weight_hh_lgstd_2 = nn.Parameter(torch.rand(gate_size, hidden_size))
#        self.weight_ih_lgstd_2 = nn.Parameter(torch.rand(gate_size, input_size))
#        self.bias_hh_lgstd_2 = nn.Parameter(torch.rand(gate_size))
#        self.bias_ih_lgstd_2 = nn.Parameter(torch.rand(gate_size))
#
#        self._all_weights = [k for k, v in self.__dict__.items() if '_ih' in k or '_hh' in k]
#        self.reset_parameters()
#
#    def extra_repr(self):
#        s = '{input_size}, {hidden_size}'
#        return s.format(**self.__dict__)
#
#    def reset_parameters(self):
#        stdv = 1.0 / math.sqrt(self.hidden_size)
#        init.uniform_(self.weight_ih_mean_1, -stdv, stdv)
#        init.uniform_(self.weight_hh_mean_1, -stdv, stdv)
#        init.uniform_(self.bias_hh_mean_1, -stdv, stdv)
#        init.uniform_(self.bias_ih_mean_1, -stdv, stdv)
#
#        init.uniform_(self.weight_ih_mean_2, -stdv, stdv)
#        init.uniform_(self.weight_hh_mean_2, -stdv, stdv)
#        init.uniform_(self.bias_hh_mean_2, -stdv, stdv)
#        init.uniform_(self.bias_ih_mean_2, -stdv, stdv)
#
#        init.uniform_(self.weight_hh_lgstd_1, 2 * math.log(stdv), math.log(stdv))
#        init.uniform_(self.weight_ih_lgstd_1, 2 * math.log(stdv), math.log(stdv))
#        init.uniform_(self.bias_hh_lgstd_1, 2 * math.log(stdv), math.log(stdv))
#        init.uniform_(self.bias_ih_lgstd_1, 2 * math.log(stdv), math.log(stdv))
#        init.uniform_(self.weight_hh_lgstd_2, 2 * math.log(stdv), math.log(stdv))
#        init.uniform_(self.weight_ih_lgstd_2, 2 * math.log(stdv), math.log(stdv))
#        init.uniform_(self.bias_hh_lgstd_2, 2 * math.log(stdv), math.log(stdv))
#        init.uniform_(self.bias_ih_lgstd_2, 2 * math.log(stdv), math.log(stdv))
#
#    def sample_weight_diff(self):
#        if self.training and self.sample:
#            weight_hh_std = torch.exp(self.weight_hh_lgstd_1)
#            epsilon = weight_hh_std.new_zeros(*weight_hh_std.size()).normal_()
#            weight_hh_diff = epsilon * weight_hh_std
#
#            weight_ih_std = torch.exp(self.weight_ih_lgstd_1)
#            epsilon = weight_ih_std.new_zeros(*weight_ih_std.size()).normal_()
#            weight_ih_diff = epsilon * weight_ih_std
#
#            bias_hh_std = torch.exp(self.bias_hh_lgstd_1)
#            epsilon = bias_hh_std.new_zeros(*bias_hh_std.size()).normal_()
#            bias_hh_diff = epsilon * bias_hh_std
#
#            bias_ih_std = torch.exp(self.bias_ih_lgstd_1)
#            epsilon = bias_ih_std.new_zeros(*bias_ih_std.size()).normal_()
#            bias_ih_diff = epsilon * bias_ih_std
#
#            weight_hh_std = torch.exp(self.weight_hh_lgstd_2)
#            epsilon = weight_hh_std.new_zeros(*weight_hh_std.size()).normal_()
#            weight_hh_diff_2 = epsilon * weight_hh_std
#
#            weight_ih_std = torch.exp(self.weight_ih_lgstd_2)
#            epsilon = weight_ih_std.new_zeros(*weight_ih_std.size()).normal_()
#            weight_ih_diff_2 = epsilon * weight_ih_std
#
#            bias_hh_std = torch.exp(self.bias_hh_lgstd_2)
#            epsilon = bias_hh_std.new_zeros(*bias_hh_std.size()).normal_()
#            bias_hh_diff_2 = epsilon * bias_hh_std
#
#            bias_ih_std = torch.exp(self.bias_ih_lgstd_2)
#            epsilon = bias_ih_std.new_zeros(*bias_ih_std.size()).normal_()
#            bias_ih_diff_2 = epsilon * bias_ih_std
#
#            return weight_hh_diff, weight_ih_diff, bias_hh_diff, bias_ih_diff, weight_hh_diff_2, weight_ih_diff_2, bias_hh_diff_2, bias_ih_diff_2
#        return 0, 0, 0, 0, 0, 0, 0, 0
#
#    def flat_parameters(self):
#        probs = F.softmax(self.weights, dim=-1)
##        print(probs)
#
#        weight_hh_1 = self.weight_hh_mean_1 * 1.
#        weight_ih_1 = self.weight_ih_mean_1 * 1.
#        bias_hh_1 = self.bias_hh_mean_1 * 1.
#        bias_ih_1 = self.bias_ih_mean_1 * 1.
#
#        weight_hh_2 = self.weight_hh_mean_2 * 1.
#        weight_ih_2 = self.weight_ih_mean_2 * 1.
#        bias_hh_2 = self.bias_hh_mean_2 * 1.
#        bias_ih_2 = self.bias_ih_mean_2 * 1.
#
#        weight_hh_1_copy = torch.zeros(weight_hh_1.size()).to(weight_hh_1.device)
#        weight_ih_1_copy = torch.zeros(weight_ih_1.size()).to(weight_ih_1.device)
#        bias_hh_1_copy = torch.zeros(bias_hh_1.size()).to(bias_hh_1.device)
#        bias_ih_1_copy = torch.zeros(bias_ih_1.size()).to(bias_ih_1.device)
#        weight_hh_2_copy = torch.zeros(weight_hh_2.size()).to(weight_hh_2.device)
#        weight_ih_2_copy = torch.zeros(weight_ih_2.size()).to(weight_ih_2.device)
#        bias_hh_2_copy = torch.zeros(bias_hh_2.size()).to(bias_hh_2.device)
#        bias_ih_2_copy = torch.zeros(bias_ih_2.size()).to(bias_ih_2.device)
#
#        weight_hh_diff, weight_ih_diff, bias_hh_diff, bias_ih_diff, weight_hh_diff_2, weight_ih_diff_2, bias_hh_diff_2, bias_ih_diff_2 = self.sample_weight_diff()
#
#        for gate in range(4):
#            weight_hh_1_bayes = weight_hh_1[gate * self.hidden_size:(gate + 1) * self.hidden_size] + \
#                                weight_hh_diff[gate * self.hidden_size:(gate + 1) * self.hidden_size] if self.sample else weight_hh_1[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            weight_hh_1_standard = weight_hh_1[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            weight_hh_1_search = weight_hh_1_bayes * probs[0][gate][1] + weight_hh_1_standard * probs[0][gate][0]
#            weight_hh_1_copy[gate * self.hidden_size:(gate + 1) * self.hidden_size] = weight_hh_1_search
#
#            weight_ih_1_bayes = weight_ih_1[gate * self.hidden_size:(gate + 1) * self.hidden_size] + \
#                                weight_ih_diff[gate * self.hidden_size:(gate + 1) * self.hidden_size] if self.sample else weight_ih_1[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            weight_ih_1_standard = weight_ih_1[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            weight_ih_1_search = weight_ih_1_bayes * probs[0][gate][1] + weight_ih_1_standard * probs[0][gate][0]
#            weight_ih_1_copy[gate * self.hidden_size:(gate + 1) * self.hidden_size] = weight_ih_1_search
#
#            bias_hh_1_bayes = bias_hh_1[gate * self.hidden_size:(gate + 1) * self.hidden_size] + \
#                              bias_hh_diff[gate * self.hidden_size:(gate + 1) * self.hidden_size] if self.sample else bias_hh_1[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            bias_hh_1_standard = bias_hh_1[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            bias_hh_1_search = bias_hh_1_bayes * probs[0][gate][1] + bias_hh_1_standard * probs[0][gate][0]
#            bias_hh_1_copy[gate * self.hidden_size:(gate + 1) * self.hidden_size] = bias_hh_1_search
#
#            bias_ih_1_bayes = bias_ih_1[gate * self.hidden_size:(gate + 1) * self.hidden_size] + \
#                              bias_ih_diff[gate * self.hidden_size:(gate + 1) * self.hidden_size] if self.sample else bias_ih_1[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            bias_ih_1_standard = bias_ih_1[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            bias_ih_1_search = bias_ih_1_bayes * probs[0][gate][1] + bias_ih_1_standard * probs[0][gate][0]
#            bias_ih_1_copy[gate * self.hidden_size:(gate + 1) * self.hidden_size] = bias_ih_1_search
#
#            weight_hh_2_bayes = weight_hh_2[gate * self.hidden_size:(gate + 1) * self.hidden_size] + \
#                                weight_hh_diff_2[gate * self.hidden_size:(gate + 1) * self.hidden_size] if self.sample else weight_hh_2[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            weight_hh_2_standard = weight_hh_2[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            weight_hh_2_search = weight_hh_2_bayes * probs[1][gate][1] + weight_hh_2_standard * probs[1][gate][0]
#            weight_hh_2_copy[gate * self.hidden_size:(gate + 1) * self.hidden_size] = weight_hh_2_search
#
#            weight_ih_2_bayes = weight_ih_2[gate * self.hidden_size:(gate + 1) * self.hidden_size] + \
#                                weight_ih_diff_2[gate * self.hidden_size:(gate + 1) * self.hidden_size] if self.sample else weight_ih_2[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            weight_ih_2_standard = weight_ih_2[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            weight_ih_2_search = weight_ih_2_bayes * probs[1][gate][1] + weight_ih_2_standard * probs[1][gate][0]
#            weight_ih_2_copy[gate * self.hidden_size:(gate + 1) * self.hidden_size] = weight_ih_2_search
#
#            bias_hh_2_bayes = bias_hh_2[gate * self.hidden_size:(gate + 1) * self.hidden_size] + \
#                              bias_hh_diff_2[gate * self.hidden_size:(gate + 1) * self.hidden_size] if self.sample else bias_hh_2[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            bias_hh_2_standard = bias_hh_2[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            bias_hh_2_search = bias_hh_2_bayes * probs[1][gate][1] + bias_hh_2_standard * probs[1][gate][0]
#            bias_hh_2_copy[gate * self.hidden_size:(gate + 1) * self.hidden_size] = bias_hh_2_search
#
#            bias_ih_2_bayes = bias_ih_2[gate * self.hidden_size:(gate + 1) * self.hidden_size] + \
#                              bias_ih_diff_2[gate * self.hidden_size:(gate + 1) * self.hidden_size] if self.sample else bias_ih_2[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            bias_ih_2_standard = bias_ih_2[gate * self.hidden_size:(gate + 1) * self.hidden_size]
#            bias_ih_2_search = bias_ih_2_bayes * probs[1][gate][1] + bias_ih_2_standard * probs[1][gate][0]
#            bias_ih_2_copy[gate * self.hidden_size:(gate + 1) * self.hidden_size] = bias_ih_2_search
#
#        return [weight_ih_1_copy[:, :].contiguous(), weight_hh_1_copy[:, :].contiguous(),
#                bias_ih_1_copy[:].contiguous(), bias_hh_1_copy[:].contiguous(),
#                weight_ih_2_copy[:, :].contiguous(), weight_hh_2_copy[:, :].contiguous(),
#                bias_ih_2_copy[:].contiguous(), bias_hh_2_copy[:].contiguous()]
#
#    def kl_divergence(self):
#        kl = 0
#
#        if self.sample:
#            weight_mean = torch.cat([self.weight_hh_mean_1, self.weight_ih_mean_1], -1)
#            weight_lgstd = torch.cat([self.weight_hh_lgstd_1, self.weight_ih_lgstd_1], -1)
#            bias_mean = torch.cat([self.bias_hh_mean_1, self.bias_ih_mean_1], -1)
#            bias_lgstd = torch.cat([self.bias_hh_lgstd_1, self.bias_ih_lgstd_1], -1)
#            weight_mean += torch.cat([self.weight_hh_mean_2, self.weight_ih_mean_1], -1)
#            weight_lgstd += torch.cat([self.weight_hh_lgstd_2, self.weight_ih_lgstd_1], -1)
#            bias_mean += torch.cat([self.bias_hh_mean_2, self.bias_ih_mean_1], -1)
#            bias_lgstd += torch.cat([self.bias_hh_lgstd_2, self.bias_ih_lgstd_1], -1)
#            pass
#        else:
#            weight_lgstd, weight_mean, bias_lgstd, bias_mean = 0., 0., 0., 0.
#        pass
#
#        kl += torch.mean(
#            weight_mean ** 2. - weight_lgstd * 2. + torch.exp(weight_lgstd * 2)) / 2.  # Max uses mean in orign
#        kl += torch.mean(
#            bias_mean ** 2. - bias_lgstd * 2. + torch.exp(bias_lgstd * 2)) / 2.  # Max uses mean in orign
#
#        return kl
#
#    @staticmethod
#    def permute_hidden(hx, permutation):
#        if permutation is None:
#            return hx
#        return hx[0].index_select(1, permutation), hx[1].index_select(1, permutation)
#
#    def forward(self, inputs, hx=None):  # noqa: F811
#        orig_input = inputs
#        # xxx: isinstance check needs to be in conditional for TorchScript to compile
#        if isinstance(orig_input, PackedSequence):
#            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
#            max_batch_size = batch_sizes[0]
#            max_batch_size = int(max_batch_size)
#        else:
#            batch_sizes = None
#            max_batch_size = inputs.size(1)
#            sorted_indices = None
#            unsorted_indices = None
#
#        if hx is None:
#            zeros = torch.zeros(self.num_layers,
#                                max_batch_size, self.hidden_size,
#                                dtype=inputs.dtype, device=inputs.device)
#            hx = (zeros, zeros)
#            pass
#        else:
#            # Each batch of the hidden state should match the input sequence that
#            # the user believes he/she is passing in.
#            hx = self.permute_hidden(hx, sorted_indices)
#            pass
#        pass
#
#        # self.flatten_parameters()
#        # print(self.flat_parameters()[0].size())
#        if batch_sizes is None:
#            result = _rnn_impls['LSTM'](inputs, hx, self.flat_parameters(), self.bias, self.num_layers,
#                                        0., self.training, False, False)
#            pass
#        else:
#            result = _rnn_impls['LSTM'](inputs, batch_sizes, hx, self.flat_parameters(), self.bias,
#                                        self.num_layers, 0., self.training, False)
#            pass
#        pass
#
#        output = result[0]
#        hidden = result[1:]
#        # xxx: isinstance check needs to be in conditional for TorchScript to compile
#        if isinstance(orig_input, PackedSequence):
#            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
#            return output_packed, self.permute_hidden(hidden, unsorted_indices)
#        else:
#            return output, self.permute_hidden(hidden, unsorted_indices)
#
