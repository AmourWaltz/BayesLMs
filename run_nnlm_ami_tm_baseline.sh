#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2020  Ke Li

# This script trains an RNN (including LSTM and GRU) or Transformer-based language model with PyTorch and performs N-best rescoring
stage=0
gpu=2
lmdata=ami+fisher # ami | ami+fisher | ami+fisher+swbd
LM=ami_fsh.o3g.kn.pr1-7 # 4gram ami.o3g.kn.pr1-7
mic=ihm
#ac_model_dir=exp/chain/tdnn7q_sp
#ac_model_dir=data/pytorchnn_ami/rescore/exp/$mic
ac_model=tdnn_with_fisher # with out fisher
#decode_dir_suffix=
#pytorch_path=exp/pytorch_lstm_bz32_hdim1024_ami+fisher+swbd
#pytorch_path=exp/pytorch_lstm_bz32_hdim1024_ami
#nn_model=$pytorch_path/model.pt
model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=512 # 512 for Transformer (to reproduce the perplexities and WERs above)
hidden_dim=512 # 512 for Transformer
nlayers=6 # 6 for Transformer
nhead=8 # for Transformer
learning_rate=0.1 # 0.1 for Transformer
seq_len=100
uncertainty=Gaussian # none for baseline, options: Bayesian, Gaussian
T_bayes_pos=FFN # bayes position, options:none, FFN, MHA, EMB
T_gauss_pos=1 # 0: none, 1: FFN
prior_path=steps/pytorchnn/prior/transformer # load pretrained prior model
prior=False # using pretrained model or not
mark=marks # save_path disctinct to uncover
inter_flag=0
inter_alpha=0.8

##################################################################################################
dropout=0.2 # baseline 0.2 | bayesian initial 0.0
##################################################################################################
itpr=0.8
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

set -e
export CUDA_VISIBLE_DEVICES=$gpu
ac_model_dir=data/pytorchnn_ami/rescore/exp/$mic
if [ "$uncertainty" == "Bayesian" ]; then
    pytorch_path=exp/pytorch-${model_type}-emb${embedding_dim}_hid${hidden_dim}_nly${nlayers}-${lmdata}-${dropout}-${uncertainty}-${T_bayes_pos}-pre${prior}-${mark}
    nn_model=$pytorch_path/model.pt
    data_dir=data/pytorchnn_ami/$lmdata
    decode_dir_suffix=pytorch-${model_type}-emb${embedding_dim}_hid${hidden_dim}_nly${nlayers}-${lmdata}-${dropout}-${uncertainty}-${T_bayes_pos}-pre${prior}-${mark}-itpr${itpr}-ib${inter_flag}-${inter_alpha}
elif [ "$uncertainty" == "Gaussian" ]; then
    pytorch_path=exp/pytorch-${model_type}-emb${embedding_dim}_hid${hidden_dim}_nly${nlayers}-${lmdata}-${dropout}-${uncertainty}-GP${T_gauss_pos}-pre${prior}-${mark}
    nn_model=$pytorch_path/model.pt
    data_dir=data/pytorchnn_ami/$lmdata
    decode_dir_suffix=pytorch-${model_type}-emb${embedding_dim}_hid${hidden_dim}_nly${nlayers}-${lmdata}-${dropout}-${uncertainty}-GP${T_gauss_pos}-pre${prior}-${mark}-itpr${itpr}-ib${inter_flag}-${inter_alpha}
else
    pytorch_path=exp/pytorch-${model_type}-emb${embedding_dim}_hid${hidden_dim}_nly${nlayers}-${lmdata}-${dropout}-${uncertainty}-GP${T_gauss_pos}-pre${prior}-${mark}
    nn_model=$pytorch_path/model.pt
    data_dir=data/pytorchnn_ami/$lmdata
    #decode_dir_suffix=pytorch-${model_type}-emb${embedding_dim}_hid${hidden_dim}_nly${nlayers}-${lmdata}-${dropout}-${uncertainty}-${T_bayes_pos}-pre${prior}-${mark}-itpr${itpr}
    decode_dir_suffix=pytorch-${model_type}-emb${embedding_dim}_hid${hidden_dim}_nly${nlayers}-${lmdata}-${dropout}-${uncertainty}-GP${T_gauss_pos}-pre${prior}-${mark}-itpr${itpr}-ib${inter_flag}-${inter_alpha}
    #data_dir=data/pytorchnn_ami/ami
fi

#mkdir -p $data_dir
mkdir -p $pytorch_path
# Check if PyTorch is installed to use with python
if python steps/pytorchnn/check_py.py 2>/dev/null; then
  echo PyTorch is ready to use on the python side. This is good.
else
  echo PyTorch not found on the python side.
  echo Please install PyTorch first. For example, you can install it with conda:
  echo "conda install pytorch torchvision cudatoolkit=10.2 -c pytorch", or
  echo with pip: "pip install torch torchvision". If you already have PyTorch
  echo installed somewhere else, you need to add it to your PATH.
  echo Note: you need to install higher version than PyTorch 1.1 to train Transformer models
  exit 1
fi

#if [ $stage -le 0 ]; then
#  local/pytorchnn/data_prep.sh $data_dir
#fi

if [ $stage -le 1 ]; then
  # Train a PyTorch neural network language model.
  echo "Start neural network language model training."
  #$cuda_cmd $pytorch_path/log/train.log utils/parallel/limit_num_gpus.sh \
	# XBY 2.20: uncertainty
    python steps/pytorchnn/train.py --data $data_dir \
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
            --epoch 64 \
            --save $nn_model \
            --prior $prior \
            --prior_path $prior_path \
            --uncertainty $uncertainty \
            --T_bayes_pos $T_bayes_pos \
            --T_gauss_pos $T_gauss_pos \
            --tied \
            --cuda > $pytorch_path/train.log
fi

#LM=ami_fsh.o3g.kn.pr1-7 # Using the 4-gram const arpa file as old lm
#LM=ami.o3g.kn.pr1-7
if [ $stage -le 2 ]; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir with a PyTorch trained $model_type LM."
  for decode_set in dev eval; do
      decode_dir=${ac_model_dir}/$ac_model/decode_${decode_set}
      steps/pytorchnn/lmrescore_nbest_pytorchnn_cuda.sh \
        --stage 1 \
        --cmd "$decode_cmd --mem 4G" \
        --N 20 \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --T_bayes_pos $T_bayes_pos \
        --T_gauss_pos $T_gauss_pos \
        --interpolation_flag $inter_flag \
        --inter_alpha $inter_alpha \
        $itpr data/pytorchnn_ami/rescore/lang_comb/lang_$LM $nn_model $data_dir/words.txt \
        data/pytorchnn_ami/rescore/data/$mic/${decode_set}_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}
  done
fi
exit 0

#  for i in data/pytorchnn_ami/rescore/exp/ihm/tdnnli_sp_bi/ihm/decode_dev; do grep Sum $i/*sco*/*ys | ./utils/best_wer.sh ;done

