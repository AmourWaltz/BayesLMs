# BayesLMs

Configuration: 

for all langauge models, 

set uncertainty = Gaussian;

for LSTM LMs, change L_gauss_pos to 1-7 for different GP activation positions in 1-layer LSTM. Set deterministic=True for fine-tuning pretrained model.

|             | baseline | input_gate | forget_gate | cell_gate | output_gate | cell_states | hidden_states | inputs | cell_gate (deterministic=True) |
| ----------- | -------- | ---------- | ----------- | --------- | ----------- | ----------- | ------------- | ------ | ------------------------------ |
| L_gauss_pos | 0        | 1          | 2           | 3         | 4           | 5           | 6             | 7      | 8                              |

for Transformer LMs, change T_gauss_pos to 1 for GP activation position in 1-layer FFN.

|             | baseline | FFN        |
| ----------- | -------- | ---------- |
| T_gauss_pos | 0        | 1          |

Best settings for training Bayesian LSTM and Transformer language models, which is similar for GPact LMs.

|             | embedding_dim | hidden_dim | nlayers | learning_rate    | dropout | pretrain | Bayesian_pos                             |
| ----------- | ------------- | ---------- | ------- | ---------------- | ------- | -------- | ---------------------------------------- |
| LSTM        | 1024          | 1024       | 2       | 5 (fine-tune=0.1)| 0.2     | False    | cell gate (L_bayes_pos=3, L_gauss_pos=3) |
| Transformer | 512           | 4096       | 6       | 0.1              | 0.2     | True     | FFN (T_bayes_pos=FFN, T_gauss_pos=3)     |

Training steps for GPact LSTM:

Pretrain a no variance LSTM (deterministic=True and no lgstd, not the baseline LSTM, only for GPact on cell gate) and then fine-tune it, Parameter mark is to annotate the model with any value and know from other models if there's any change of the code.:
```
 bash run_nnlm_ami_lstm_baseline.sh --L_gauss_pos 8 --mark no # Pretrain the LSTM without lgstd. 
 cp path/pretrained_model.pt steps/pytorchnn/prior/lstm/ # copy the pretrained model to specific path.
 bash run_nnlm_ami_lstm_baseline.sh --L_gauss_pos 3 --learning_rate 0.1 --prior True --mark no # Fine-tune the pretrained model
 
```

Training steps for GPact Transformer:

Do GPact learning directly on 1-layer feed-forward network:
```
 bash run_nnlm_ami_tm_baseline.sh --hidden_dim 4096 --T_gauss_pos 1 --mark no
 
```

Lattice rescore:

Noticed to replace oringinal by lmrescore_nbest_pytorchnn_cuda.sh.

```
 steps/pytorchnn/lmrescore_nbest_pytorchnn_cuda.sh
```

Replace other .sh files in specific paths as the number of nj is set to 5 in rescoring step if necessary.

```
 local/score_asclite.sh
 bash data/pytorchnn_ami/rescore/exp/mdm8/tdnn_with_fisher/decode_dev/cat_lat.sh

```

