#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2020  Ke Li

# This script trains an RNN (including LSTM and GRU) or Transformer-based language model with PyTorch and performs N-best rescoring

# Dev/eval2000 perplexity of a 2-layer LSTM model is: 47.1/41.9. WERs with N-best rescoring (with hidden states carried over sentences):
# %WER 10.9 | 4459 42989 | 90.5 6.4 3.1 1.4 10.9 42.7 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm_nbest//score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 7.1 | 1831 21395 | 93.8 4.1 2.1 0.9 7.1 36.4 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm_nbest//score_11_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.6 | 2628 21594 | 87.3 8.5 4.1 1.9 14.6 46.7 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm_nbest//score_10_0.0/eval2000_hires.ctm.callhm.filt.sys
# Without hidden-state-carry-over, the WER on eval2000 from N-best rescoring with the LSTM model is 11.2

# Dev/eval2000 perplexity of a Transformer LM is: 47.0/41.6. WERs with N-best rescoring:
# %WER 10.8 | 4459 42989 | 90.6 6.3 3.1 1.5 10.8 42.1 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer_nbest//score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 7.2 | 1831 21395 | 93.7 4.2 2.1 1.0 7.2 37.3 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer_nbest//score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.4 | 2628 21594 | 87.6 8.3 4.1 2.0 14.4 45.5 | exp/chain/tdnn7q_sp/decode_eval2000_sw1_fsh_fg_pytorch_transformer_nbest//score_10_0.0/eval2000_hires.ctm.callhm.filt.sys

# Begin configuration section.
source /opt/share/etc/gcc-5.4.0.sh
pytorch_path=exp/transformer-swbd
export CUDA_VISIBLE_DEVICES=2
model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=512 # 512 for Transformer (to reproduce the perplexities and WERs above)
hidden_dim=4096 # 512 for Transformer
nlayers=6 # 6 for Transformer
nhead=8 # for Transformer
learning_rate=0.1 # 0.1 for Transformer
seq_len=100
dropout=0.1

data_dir=/Users/collcertaye/WorkSpace/Speech_Recognition/dataset/fast
echo "Start neural network language model training."
python train.py --data $data_dir \
        --model $model_type \
        --emsize $embedding_dim \
        --nhid $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --lr $learning_rate \
        --dropout $dropout \
        --seq_len $seq_len \
        --clip 1.0 \
        --batch-size 32 \
        --epoch 1 \
        --tied \
        #--cuda
