from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from sqlite3 import SQLITE_DROP_TEMP_TABLE
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os

import data
import model
import numpy as np
from utils import create_exp_dir, get_logger



###############################################################################
# Load the model
###############################################################################

#prior, _, _ = torch.load(os.path.join(args.prior_path, 'model.pt'), map_location=torch.device('cpu'))
with open(os.path.join("exp/pytorch-LSTM-emb1024_hid1024_nly2-ami+fisher-0.2-Bayesian-3-preTrue-base-1set/", 'model.pt'), 'rb') as f:
    prior_dict = torch.load(f, map_location=lambda storage, loc: storage)
pass
#print(prior.state_dict().keys())
# model_dict = model.state_dict()
# prior_dict =  {k: v for k, v in prior_dict.items() if k in model_dict}
#print(model_dict['transformerlayers.0.gpnn.coef_mean'].mean(dim=1))

for k, v in prior_dict.items():
    # print("out: ", k)
    if "rnn.weight_hh_lgstd_1" in k:
        # prior_dict = {k: v}
        print("key: ", k)
        print("value: ", v)
        # print("avg_var: ", np.exp(v)+1)
        sigma = np.log(np.exp(v)+1)
        pass
    if "rnn.weight_hh_mean_1" in k:
        print("key: ", k)
        print("value: ", v)
        # print("avg_var: ", np.abs(v))
        mean = np.abs(v[(3 - 1) * 1024:3 * 1024])
        # mean = np.abs(v)
        # sigma = np.log(np.exp(v)+1)
        pass
    # else:
    #     print("out: ", k)
    # pass
pass

snr = mean/sigma
# np.save('snr_file/trans_1s.npy', snr)

print("SNR: ", snr.median())
# model_dict.update(prior_dict)
# model.load_state_dict(model_dict)

