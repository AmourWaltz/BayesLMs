# BayesianLM

Configuration: 

for all langauge models, 

set uncertainty = Gaussian;

for LSTM LMs, change L_gauss_pos to 1-7 for different GP activation positions in 1-layer LSTM.

|             | baseline | input_gate | forget_gate | cell_gate | output_gate | cell_states | hidden_states | inputs |
| ----------- | -------- | ---------- | ----------- | --------- | ----------- | ----------- | ------------- | ------ |
| L_gauss_pos | 0        | 1          | 2           | 3         | 4           | 5           | 6             | 7      |


Ex:

```
 bash run_nnlm_ami_lstm_baseline.sh --L_gauss_pos 1
```

Noticed to replace oringinal by lmrescore_nbest_pytorchnn_cuda.sh.

```
 steps/pytorchnn/lmrescore_nbest_pytorchnn_cuda.sh
```

Replace other .sh files in specific paths as the number of nj is set to 5 in rescoring step if necessary.

```
 local/score_asclite.sh
 bash data/pytorchnn_ami/rescore/exp/mdm8/tdnn_with_fisher/decode_dev/cat_lat.sh

```

