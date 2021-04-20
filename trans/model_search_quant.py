import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from genotypes import Genotype
from torch.autograd import Variable
from collections import namedtuple
# from model import DARTSCell, RNNModel
from quantModel import QuantizeLinear, QuantMultiheadAttention, PositionalEncoding

INITRANGE = 0.04
nbits = [1, 2, 4, 8]

class QuantTransEncLDARTSCell(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
      super(QuantTransEncLDARTSCell, self).__init__()
      self.dropout = nn.Dropout(dropout)
      self.norm1 = nn.LayerNorm(d_model)
      self.norm2 = nn.LayerNorm(d_model)
      self.dropout1 = nn.Dropout(dropout)
      self.dropout2 = nn.Dropout(dropout)

      self.activation = nn.GELU()

      #self-attention layer
      self.self_attn = []
      for nbit in nbits:
          self.self_attn.append(
              QuantMultiheadAttention(d_model, nhead, dropout=dropout, nbit=nbit).cuda()
            )

      self.linear1 = []
      for nbit in nbits:
          self.linear1.append(
              QuantizeLinear(d_model, dim_feedforward, nbit).cuda()
            )

      self.linear2 = []
      for nbit in nbits:
          self.linear2.append(
              QuantizeLinear(dim_feedforward, d_model, nbit).cuda()
            )

    def forward(self, src, src_mask=None):
      probs = F.softmax(self.weights, dim=-1)
      
      src2 = torch.zeros(src.size()).cuda()
      for i, sa_l in enumerate(self.self_attn):
          src2 += sa_l(src, src, src, attn_mask=src_mask)[0] * probs[0][i]
      src = src + self.dropout1(src2)
      src = self.norm1(src)

      src2 = torch.zeros(src.size()).cuda()
      for i, li_l in enumerate(self.linear1):
          src2 += li_l(src) * probs[1][i]
      src2 = self.dropout(self.activation(src2))

      src3 = torch.zeros(src.size()).cuda()
      for i, li_l in enumerate(self.linear2):
          src3 += li_l(src2) * probs[2][i]
      src = src + self.dropout2(src3)
      src = self.norm2(src)
      return src

class QuantTransformerModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, tie_weights=False,
                 cell_cls=QuantTransEncLDARTSCell, genotype=None):
        super(QuantTransformerModel, self).__init__()
        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except ImportError:
        #     raise ImportError('TransformerEncoder module does not exist in '
        #                       'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformerlayers = nn.ModuleList()

        if cell_cls == QuantTransEncLDARTSCell:
            assert genotype is None
            for i in range(nlayers):
                self.transformerlayers.append(
                    cell_cls(ninp, nhead, nhid, dropout).cuda()
                )
        else:
            assert genotype is not None
            for i in range(nlayers):
                self.transformerlayers.append(
                    cell_cls(ninp, nhead, nhid, dropout, genotype)
                )

        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal '
                                 'to emsize.')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.nlayers = nlayers
        self.cell_cls = cell_cls

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
                mask == 1, float(0.0))
        return mask.cuda()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).cuda()
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        # output = self.transformerlayers(src, self.src_mask)
        output = src 
        for mod in self.transformerlayers:
            output = mod(output, src_mask=self.src_mask)

        output = self.decoder(output)
        return output

class QuantTransModelSearch(QuantTransformerModel):
    """docstring for QuantTransModelSearch"""
    def __init__(self, *args):
        super(QuantTransModelSearch, self).__init__(*args, cell_cls=QuantTransEncLDARTSCell, genotype=None)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = QuantTransModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
        weights_data = torch.randn(self.nlayers, 3, 4).mul_(1e-3)
        self.weights = Variable(weights_data.cuda(), requires_grad=True)
        self._arch_parameters = [self.weights]
        for i, tran_l in enumerate(self.transformerlayers):
            tran_l.weights = self.weights[i]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(probs):
            gene = []
            for i in range(self.nlayers):
                W = probs[i].copy() # 3 * 4 
                j = W.argmax(dim=-1)

                gene.append([nbits[i] for i in j])
            return gene

        gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
        genotype = Genotype(transformer=gene)
        return genotype

    

