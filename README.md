# BayesLMs

This repository contains the experimental settings for the paper "[Bayesian Neural Network Language Modeling for Speech Recognition](https://ieeexplore.ieee.org/abstract/document/9874985)". The code is based on the Kaldi recipe and mainly implemented using PyTorch. 

If you need to refer to our resource, please cite our work with the bibtex listed blow:
```bibtext
@ARTICLE{9874985,
  author={Xue, Boyang and Hu, Shoukang and Xu, Junhao and Geng, Mengzhe and Liu, Xunying and Meng, Helen},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Bayesian Neural Network Language Modeling for Speech Recognition}, 
  year={2022},
  volume={30},
  number={},
  pages={2900-2917},
  doi={10.1109/TASLP.2022.3203891}}
```
A single V100 GPU card is used for training. The detailed configurations and the training process for Bayesian LMs are presented as follows

## Prerequisite:

Please run the following command and install the packages.
```shell
 pip install -r requirements.txt
```

## Implementation:

### Model Parameters

|             | embedding_dim | hidden_dim | nlayers | learning_rate         | dropout  |
| ----------- | ------------- | ---------- | ------- | --------------------- | ------- |
| LSTM        | 1024          | 1024       | 2       | 5 (fine-tune=0.1)     | 0.2      |
| Transformer | 512           | 4096       | 6       | 0.1 (fine-tune=0.01) | 0.2    |


### Training Steps for Bayesian, GP and Variational Transformer LMs

1. Train the baseline Transformer LM:
```shell
 bash run_nnlm_ami_tm.sh --uncertainty none --mark no
```

2. Pretrain a standard Transformer LM using our self-built Transformers. Note that our self-built standard Transformer is comparable to the baseline Transformer in step 1:
```shell
 bash run_nnlm_ami_tm.sh --uncertainty Bayesian --T_bayes_pos none --mark no
```

3. Fine-tune the Bayesian Transformer on the pretrained model:
```shell
 # Copy the pretrained model under the specified path.
 cp path/pretrained_model.pt steps/pytorchnn/prior/transformer/

 # Fine-tune the Bayesian Transformer LM.
 # Parameter "T_bayes_pos" denotes the selected network internal positions to be Bayesian estimated as discussed in Fig. 2 in paper.
 # [ T_bayes_pos: FFN - feed-forward network | EMB - word embedding layer | MHA - multi-head attention module ]
 bash run_nnlm_ami_tm.sh --uncertainty Bayesian --T_bayes_pos FFN --learning_rate 0.01 --prior True --mark no
```

&emsp;Practically, we can stop the training process in previous and fine-tune on the half-converged pretrained standard model as described in paper. Experimental results suggested that the model parameters initialized from the half-converged or fully converged pretrained standard models produced similar performance, while both marginally better than using random weights.

4. For GP Transformer, we only need to change the values of some parameters and fine-tune in the same way as Bayesian Transformer:
```shell
 # Pretrain a standard Transformer LM using our self-built Transformers.
 bash run_nnlm_ami_tm.sh --uncertainty Gaussian --T_gauss_pos 0 --mark no

 # Copy the pretrained model under the specified path.
 cp path/pretrained_model.pt steps/pytorchnn/prior/transformer/

 # Fine-tune the GP Transformer LM.
 # Parameter "T_gauss_pos" denotes the selected network internal positions and GP types for activation coefficients and parameters to be Bayesian estimated as discussed in Fig. 2 in paper.
 bash run_nnlm_ami_tm.sh --uncertainty Gaussian --T_gauss_pos 3 --learning_rate 0.01 --prior True --mark no
```

&emsp;Different types that separately model the uncertainty over model parameters $ \theta $ and the coefficients of hidden neural activation functions $ \lambda $. We have compared the performance of several ways and finally choose to set "T_gauss_pos" to 3 in this work.
| $\lambda$ | $\theta$ | T_gauss_pos |
| ------- | ------- | -------------- |
| &times; | &times; | 0              |
| &radic; | &times; | 1              |
| &times; | &radic; | 2              |
| &radic; | &radic; | 3              |

5. For Variational Transformer, we only need to change the values of some parameters and fine-tune in the same way as Bayesian Transformer:
```shell
 # Pretrain a standard Transformer LM using our self-built Transformers.
 bash run_nnlm_ami_tm.sh --uncertainty Variational --T_v_pos 00 --mark no

 # Copy the pretrained model under the specified path.
 cp path/pretrained_model.pt steps/pytorchnn/prior/transformer/

 # Fine-tune the Variational Transformer LM.
 # Parameter "T_v_pos" denotes the selected network internal positions to be Bayesian estimated as discussed in Fig. 2 in paper.
 # [T_v_pos: 01 - apply VNN on the 1st Transformer layer after to the word embedding | 10 - apply VNN on the 2nd Transformer layer | 11 - apply VNN on both the 1st and 2nd Transformer layer ]
 bash run_nnlm_ami_tm.sh --uncertainty Variational --T_v_pos 11 --learning_rate 0.01 --prior True --mark no
```


### Training Steps for Bayesian, GP and Variational LSTM-RNN LMs

1. Train the baseline LSTM-RNN LM:
```shell
 bash run_nnlm_ami_lstm.sh --uncertainty none --mark no
```

2. Pretrain a standard LSTM-RNN LM using our self-built LSTM-RNNs. Note that our self-built standard LSTM-RNN is comparable to the baseline LSTM-RNN in step 1:
```shell
 bash run_nnlm_ami_lstm.sh --uncertainty Bayesian --L_bayes_pos none --mark no
```

3. Fine-tune the Bayesian LSTM-RNN on the pretrained model:
```shell
 # Copy the pretrained model under the specified path.
 cp path/pretrained_model.pt steps/pytorchnn/prior/lstm/

 # Fine-tune the Bayesian LSTM-RNN LM.
 # Parameter "L_bayes_pos" denotes the selected network internal positions to be Bayesian estimated as discussed in Fig. 2 in paper.
 # [ L_bayes_pos: 1 - input gate | 2 - forget gate | 3 - cell input | 4 - output gate ]
 bash run_nnlm_ami_lstm.sh --uncertainty Bayesian --L_bayes_pos 3 --learning_rate 0.1 --prior True --mark no
```

&emsp;The default setting is to apply Bayesian estimation in both two LSTM layers. We can manually change the codes in `steps/pytorchnnpytorchnn/model.py` for only the 1st or the 2nd layer.

4. For GP LSTM-RNN, we only need to change the values of some parameters and fine-tune in the same way as Bayesian LSTM-RNN followed by step 2:
```shell
 # Pretrain a standard LSTM-RNN LM using our self-built LSTM-RNNs.
 bash run_nnlm_ami_lstm.sh --uncertainty Gaussian --L_gauss_pos 0 --mark no

 # Copy the pretrained model under the specified path.
 cp path/pretrained_model.pt steps/pytorchnn/prior/lstm/

 # Fine-tune the GP LSTM-RNN LM.
 # Parameter "L_gauss_pos" denotes the selected network internal positions and GP types for activation coefficients and parameters to be Bayesian estimated as discussed in Fig. 2 in paper.
 bash run_nnlm_ami_lstm.sh --uncertainty Gaussian --L_gauss_pos 6360 --learning_rate 0.1 --prior True --mark no
```

&emsp;For LSTM-RNN LMs, change L_gauss_pos[1] (gp_type) to 0-3 for different GP LSTM-RNN types as discussed in GP Transformer. L_gauss_pos[0]. L_gauss_pos[0] and L_gauss_pos[2] denotes different gates to be Bayesian estimated as follows and discussed in Fig. 2 and Table III in paper. "L_gauss_pos[3]=0" denotes using GP on both two layers.

|           | baseline | input gate | forget gate | cell input | output gate | cells | hidden gate | inputs |
| -------------- | -------- | ---------- | ----------- | --------- | ----------- | ----- | ------- | ------ |
| L_gauss_pos[0] | 0        | 1          | 2           | 3         | 4           | 5     | 6       | 7      |

5. For LSTM-RNN Transformer, we only need to change the values of some parameters and fine-tune in the same way as Bayesian LSTM-RNN followed by step 2:
```shell
 # Pretrain a standard LSTM-RNN LM using our self-built LSTM-RNNs.
 bash run_nnlm_ami_lstm.sh --uncertainty Variational --L_v_pos 00 --mark no

 # Copy the pretrained model under the specified path.
 cp path/pretrained_model.pt steps/pytorchnn/prior/lstm/

 # Fine-tune the Variational LSTM-RNN LM.
 # Parameter "T_v_pos" denotes the selected network internal positions to be Bayesian estimated as discussed in Fig. 2 in paper.
 # [ L_v_pos: 01 - apply VNN on the 1st layer after word embedding layer | 10 - apply VNN on the 2nd layer | 11 - apply VNN on both the 1st and 2nd layer ]
 bash run_nnlm_ami_lstm.sh --uncertainty Variational --L_v_pos 11 --learning_rate 0.1 --prior True --mark no
```


### N-best Rescoring:

N-best lists rescoring using the trained Bayesian NNLMs, for example, the Bayesian Transformer LM in step 3 and GP LSTM-RNN in step 4 after interpolation with the baseline NNLMs.

```shell
 bash run_nnlm_ami_tm.sh --stage 2 --uncertainty Bayesian --T_bayes_pos FFN --prior True --mark no --inter_flag 1

 bash run_nnlm_ami_lstm.sh --stage 2 --uncertainty Bayesian --L_gauss_pos 6360 --prior True --mark no --inter_flag 1
```

&emsp;Note that "ratio" denotes the interpolation weights of n-gram LMs.
<!-- <br> -->

### Training process of NAS:

The super-network for both Bayesian estimated Transformer and LSTM-RNN LMs have been established in `steps/pytorchnn/model_search_bayes.py`. 
Replace the `steps/pytorchnn/train.py` by `steps/pytorchnn/train_search_bayes.py` in the shells and directly run the following commands. 

```shell
 # Bayesian Transformer using NAS.
 bash run_nnlm_ami_tm.sh --uncertainty Bayesian --mark no 

 # GP Transformer using NAS.
 bash run_nnlm_ami_tm.sh --uncertainty Gaussian --mark no 

 # Bayesian LSTM-RNN using NAS.
 bash run_nnlm_ami_lstm.sh --uncertainty Bayesian --mark no

 # GP LSTM-RNN using NAS.
 bash run_nnlm_ami_lstm.sh --uncertainty Gaussian --mark no 

```

&emsp;We can directly check the search results in `path/train.log` where the architecture weights would be printed out.
<!-- <br> -->

### SNR Analysis

Change parameter *path* in `steps/pytorchnn/variance.py` by model path under `path/model.pt`. Run the following command directly and the SNR (Signal-to-Noise Ratio) results will be printed out.

```shell
python steps/pytorch/variance.py
```

## Citation

If you are interested in this work, please kindly cite [our paper](https://ieeexplore.ieee.org/abstract/document/9874985):

``` latex
@ARTICLE{9874985,
  author={Xue, Boyang and Hu, Shoukang and Xu, Junhao and Geng, Mengzhe and Liu, Xunying and Meng, Helen},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Bayesian Neural Network Language Modeling for Speech Recognition}, 
  year={2022},
  volume={30},
  number={},
  pages={2900-2917},
  keywords={Bayes methods;Transformers;Uncertainty;Computational modeling;Artificial neural networks;Computer architecture;Task analysis;Bayesian learning;model uncertainty;neural architecture search;neural language models;speech recognition},
  doi={10.1109/TASLP.2022.3203891}}
```
