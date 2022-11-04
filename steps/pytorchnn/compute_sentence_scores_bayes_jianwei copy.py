# Copyright 2020    Ke Li

""" This script computes sentence scores with a PyTorch trained neural LM.
    It is called by steps/pytorchnn/lmrescore_nbest_pytorchnn.sh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from collections import defaultdict
import math
from xml.dom.expatbuilder import parseString

import torch
import torch.nn as nn


def load_nbest(path):
    r"""Read nbest lists.

    Assume the input file format is as following:
        en_4156-A_030185-030248-1 oh yeah
        en_4156-A_030470-030672-1 well i'm going to have mine and two more classes
        en_4156-A_030470-030672-2 well i'm gonna have mine and two more classes
        ...

    Args:
        path (str): A file of nbest lists with the above format.

    Returns:
        The nbest lists represented by a dictionary from string to a list of
        strings. The key is utterance id and the value is the hypotheses.
    """

    nbest = defaultdict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            try:
                key, hyp = line.split(' ', 1)
            except ValueError:
                key = line
                hyp = ' '
            key = key.rsplit('-', 1)[0]
            if key not in nbest:
                nbest[key] = [hyp]
            else:
                nbest[key].append(hyp)
    return nbest


def read_vocab(path):
    r"""Read vocabulary.

    Args:
        path (str): A file with a word and its integer index per line.

    Returns:
        A vocabulary represented by a dictionary from string to int (starting
        from 0).
    """

    word2idx = {}
    idx2word = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.split()
            assert len(word) == 2
            word = word[0]
            if word not in word2idx:
                idx2word.append(word)
                word2idx[word] = len(idx2word) - 1
                pass
            pass
        pass

    return word2idx


def get_input_and_target(hyp, vocab, seg_wise=False, decode_set='eval2000', model_type='LSTM', seq_len=150):
    r"""Convert a sentence to lists of integers, with input and target separately.

    Args:
        hyp (str):  Sentence, with words separated by spaces, e.g. 'hello there'
        vocab:      A dictionary from string to int, e.g. {'<s>':0, 'hello':1,
                    'there':2, 'apple':3, ...}

    Returns:
        A pair of lists, one with the integerized input sequence, one with the
        integerized output/target sequence: in this case ([0, 1, 2], [1 2 0]),
        because the input sequence has '<s>' added at the start and the output
        sequence has '<s>' added at the end.
        Words that are not in the vocabulary will be converted to '<unk>', which
        is expected to be in the vocabulary if there are out-of-vocabulary words.
    """
    input_string = '<s> ' + hyp
    output_string = hyp + ' <s>'
    if seg_wise == 1 and os.path.exists("steps/pytorchnn/" + decode_set + '_' + model_type + str(seq_len) + ".txt"):
        # cur_len = len(input_string)
        with open("steps/pytorchnn/" + decode_set + '_' + model_type + str(seq_len) + ".txt", "r") as f:
            # print("file:", f.readline())
            context = f.readline()
            input_string = context[-1200:] + ' ' + input_string
            # print("input_str", len(input_string))
            pass
            # print("file:", context.count(' '))
            # if len(context) + cur_len >= 128:
            #     cut_len = 128 - cur_len
            #     input_string = context[-cut_len:] + ' ' + input_string
            #     pass
            # else:
            #     input_string = context[:] + ' ' + input_string
            #     pass
            # pass
            # print("sentence:", input_string)
        pass

    #input_string = hyp
    #output_string = hyp
    oov_num = 0
    input_ids, output_ids = [], []
    for word in input_string.split():
        try:
            input_ids.append(vocab[word]) # word.lower()
        except KeyError:
            oov_num += 1
            # with open("oov2.txt", "a") as f:
            #     f.write(word + '\n')
            #     pass
            input_ids.append(vocab['<unk>'])
    for word in output_string.split():
        try:
            output_ids.append(vocab[word]) # word.lower()
        except KeyError:
            output_ids.append(vocab['<unk>'])
            pass
        pass

    # return input_ids[0:-1], output_ids[1:]
    # print("input_ids", len(input_ids))
    return input_ids, output_ids, oov_num


def compute_sentence_score(model, criterion, ntokens, data, target,
                           model_type='LSTM', hidden_1=None, inter_flag=False, alpha=0.0, model_2=None, hidden_2=None, seg_wise=False, seq_len=128, model_var='none'):
    r"""Compute neural language model score of a sentence.

    Args:
        model:      A neural language model.
        criterion:  Training criterion of a neural language model, e.g.
                    cross entropy.
        ntokens:    Vocabulary size.
        data:       Integerized input sentence.
        target:     Integerized target sentence.
        model_type: Model type, e.g. LSTM or Transformer or others.
        hidden:     Initial hidden state for getting the score of the input
                    sentence with a recurrent-typed neural language model
                    (optional).

    Returns:
        The score (negative log-likelihood) of the input sequence from a neural
        language model. If the model is recurrent-typed, the function has an
        extra output: the last hidden state after computing the score of the
        input sequence.
    """

    # print(len(target), len(data))
    length = len(target)
    #print(data.size())
    data = torch.LongTensor(data).view(-1, 1).contiguous().cuda()
    target = torch.LongTensor(target).view(-1).contiguous().cuda()

    if model_var == 'back':
        # print('back', data.squeeze(-1).size(), target.unsqueeze(-1).size())
        target_back = torch.flip(data.squeeze(-1), [0])
        data_back = torch.flip(target.unsqueeze(-1), [0])
        # hidden_1 = model.init_hidden(1)
        pass

    if seg_wise == 1:
        if len(data) > seq_len and length < seq_len:
            data_inp = data[-seq_len:]
            # print("cross")
            pass
        else:
            data_inp = data[-length:]
            # print("internal")
        pass
    else:
        data_inp = data
        pass
    pass

    mems = tuple()
    print("sentence:", len(data_inp))
    with torch.no_grad():
        # Setp 1: Compute the probability of the main LM.
        # Noted that the input of the backward LSTM LM (model_var == 'back') is the inverted word sequence data_back.
        if model_type == 'Transformer':
            if model_var == 'none':
                output_1 = model(data_inp)
                pass
            elif model_var == 'XL':
                output_1, mems = model(data, *mems)
                pass
            pass
        else:
            if model_var == 'back':
                output_1, hidden_1 = model(data_back, hidden_1)
                pass
            else:
                output_1, hidden_1 = model(data_inp, hidden_1)
                pass
            pass

        # Step 2: Compute the probability of the interpolated LM if required.
        # inter_flag 0: no interpolated LM.
        # inter_flag 1: score = log (P_1 + P_2) in token level which is more likely to a voting strategy.
        # inter_flag 2: score = log P_1 + log P_2 in sentence level which is more likely to a smoothing strategy.
        if inter_flag != 0:
            if model_var == 'back' and seg_wise == 1:
                output_3, hidden_2 = model_2(data, hidden_2)
                output_2 = output_3[-length:]
                pass
            else:
                output_2, hidden_2 = model_2(data[-length:], hidden_2)
                output = alpha * output_1[-length:] + (1. - alpha) * output_2
                pass
            pass
        else:
            output = output_1[-length:]
            hidden_2 = None
            pass
        
        # Step 3: Select the interpolation approaches by the arg inter_flag.
        # loss_1 represents the score of the main LM and loss_2 represents the score of interpolated LM. 
        if model_var == 'back':
            # For backward LSTM LMs as the main LM, 
            # we prefer to compute the interpolated sentence score only combined with the forward LSTM LM.
            # Specifically, the loss_1 needs to be computed using the inverted backward sequence target_back. 
            if inter_flag == 2:
                loss_1 = criterion(output_1.view(-1, ntokens), target_back[:length])
                loss_2 = criterion(output_2.view(-1, ntokens), target)
                loss = alpha * loss_1 + (1. - alpha) * loss_2
                pass
            elif inter_flag == 0:
                loss = criterion(output_1.view(-1, ntokens), target_back[:length])
                pass
            pass
        else:
            if inter_flag == 1 or inter_flag == 0:
                loss = criterion(output.view(-1, ntokens), target)
                pass
            elif inter_flag == 2:
                loss_1 = criterion(output_1[-length:].view(-1, ntokens), target)
                loss_2 = criterion(output_2.view(-1, ntokens), target)
                loss = alpha * loss_1 + (1. - alpha) * loss_2
                pass
            pass
        pass
    
    sent_score = (length) * loss.item()
    # sent_score = -loss.item()
    print(sent_score)
    if model_type == 'Transformer':
        return sent_score, hidden_2
    return sent_score, hidden_1, hidden_2


def compute_scores(nbest, model, criterion, ntokens, vocab, model_type='LSTM', inter_flag=False, alpha=0.0, model_2=None, seg_wise=False, decode_set='eval2000', seq_len=128, model_var='none'):
    r"""Compute sentence scores of nbest lists from a neural language model.

    Args:
        nbest:      The nbest lists represented by a dictionary from string
                    to a list of strings.
        model:      A neural language model.
        criterion:  Training criterion of a neural language model, e.g.
                    cross entropy.
        ntokens:    Vocabulary size.
        model_type: Model type, e.g. LSTM or Transformer or others.

    Returns:
        The nbest litsts and their scores represented by a dictionary from
        string to a pair of a hypothesis and its neural language model score.
    """

    # Turn on evaluation mode which disables dropout.
    #print("enter in:", nbest)
    model.eval()
    if inter_flag == 1:
        model_2.eval()
    total_score = 0
    total_len = 0
    looper = 0
    oov_num = 0

    nbest_and_scores = defaultdict(float)
    if model_type != 'Transformer':
        hidden_1 = model.init_hidden(1)
        if inter_flag == 1:
            hidden_2 = model_2.init_hidden(1)
        else:
            hidden_2 = None
            pass
        pass

    # For Transformer interpolated with LSTM
    if model_type == 'Transformer' and inter_flag != 0:
        hidden_2 = model_2.init_hidden(1)
        pass
    else:
        hidden_2 = None
        pass

    for key in nbest.keys():
        print("key:", key)
        scores_list = []

        if model_type != 'Transformer':
            cached_hiddens_1 = []
            cached_hiddens_2 = []
            pass

        if model_type == 'Transformer' and inter_flag != 0:
            cached_hiddens_2 = []
            pass

        if seg_wise == 1 and model_type == 'LSTM':
            hidden_1 = model.init_hidden(1)
            pass

        if seg_wise == 1 and model_var == 'back':
            hidden_2 = model_2.init_hidden(1)
            pass

        for hyp in nbest[key]:
            print("hyp:", hyp)
            x, target, oov = get_input_and_target(hyp, vocab, seg_wise, decode_set, model_type, seq_len)
            oov_num += oov

            if model_type == 'Transformer':
                score, new_hidden_2 = compute_sentence_score(model, criterion, ntokens, x,
                                            target, model_type, inter_flag=inter_flag, alpha=alpha, model_2=model_2, hidden_2=hidden_2, seg_wise=seg_wise, seq_len=seq_len, model_var=model_var)
                if inter_flag != 0:
                    cached_hiddens_2.append(new_hidden_2)
                    pass
                pass
            else:
                score, new_hidden_1, new_hidden_2 = compute_sentence_score(model, criterion, ntokens, x, target,
                                                        model_type, hidden_1, inter_flag, alpha=alpha, model_2=model_2, hidden_2=hidden_2, seg_wise=seg_wise, seq_len=seq_len, model_var=model_var)
                cached_hiddens_1.append(new_hidden_1)
                if inter_flag != 0:
                    cached_hiddens_2.append(new_hidden_2)
                    pass
                pass

            total_len += len(x)
            total_score += score
            if key in nbest_and_scores:
                nbest_and_scores[key].append((hyp, score))
                scores_list.append(score)
                pass
            else:
                nbest_and_scores[key] = [(hyp, score)]
                scores_list = [score]
                pass

            looper += 1
            pass

        # 1.25: seg_wise cross-sentence
        min_id = scores_list.index(min(scores_list))
        if seg_wise == 1:
            # print(nbest_and_scores[key], looper)
            ext_sent, _ = nbest_and_scores[key][min_id]
            print(ext_sent)
            with open("steps/pytorchnn/" + decode_set + "_" + model_type + str(seq_len) + ".txt", "a") as f:
                f.write('<s> ' + ext_sent + ' ')
                pass
            pass
        # print(nbest_and_scores[key], looper)
        looper = 0

        # For RNN based LMs, initialize the current initial hidden states with
        # those from hypotheses of a preceeding previous utterance.
        # This achieves modest WER reductions compared with zero initialization
        # as it provides context from previous utterances. We observe that using
        # hidden states from which hypothesis of the previous utterance for
        # initialization almost doesn't make a difference. So to make the code
        # more general, the hidden states from the first hypothesis of the
        # previous utterance is used for initialization. You can also use those
        # from the one best hypothesis or just average hidden states from all
        # hypotheses of the previous utterance.
        if model_type != 'Transformer':
            hidden_1 = cached_hiddens_1[min_id]
            if inter_flag != 0:
                hidden_2 = cached_hiddens_2[min_id]
                pass
            pass

        if model_type == 'Transformer' and inter_flag != 0:
            hidden_2 = cached_hiddens_2[min_id]
            pass
        pass

    print(oov_num)
    print(total_len)
    print(total_score / total_len)
    # print(nbest_and_scores)

    return nbest_and_scores


def write_scores(nbest_and_scores, path):
    r"""Write out sentence scores of nbest lists in the following format:
        en_4156-A_030185-030248-1 7.98671
        en_4156-A_030470-030672-1 46.5938
        en_4156-A_030470-030672-2 46.9522
        ...

    Args:
        nbest_and_scores: The nbest lists and their scores represented by a
                          dictionary from string to a pair of a hypothesis and
                          its neural language model score.
        path (str):       A output file of nbest lists' scores in the above format.
    """

    with open(path, 'w', encoding='utf-8') as f:
        for key in nbest_and_scores.keys():
            for idx, (_, score) in enumerate(nbest_and_scores[key], 1):
                #print(score)
                current_key = '-'.join([key, str(idx)])
                f.write('%s %.4f\n' % (current_key, score))
    print("Write to %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Compute sentence scores of "
                                     "nbest lists with a PyTorch trained "
                                     "neural language model.")
    parser.add_argument('--nbest-list', type=str, required=True,
                        help="N-best hypotheses for rescoring")
    parser.add_argument('--outfile', type=str, required=True,
                        help="Output file with language model scores associated "
                        "with each hypothesis")
    parser.add_argument('--vocabulary', type=str, required=True,
                        help="Vocabulary used for training")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to a pretrained neural model.")
    parser.add_argument('--model', type=str, default='LSTM',
                        help='Network type. can be RNN, LSTM or Transformer.')
    parser.add_argument('--emsize', type=int, default=1024,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1024,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the '
                        'transformer model')
    parser.add_argument('--emsize_l', type=int, default=1024,
                        help='size of word embeddings')
    parser.add_argument('--nhid_l', type=int, default=1024,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers_l', type=int, default=2,
                        help='number of layers')

    # XBY 1.25: model variants type selection
    parser.add_argument('--model_var', type=str, default='none',
                        help='model variants type: [none]')
    parser.add_argument('--seg_wise', type=int, default=0,
                        help='model variants type: [none]')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='sequence length')
    parser.add_argument('--decode_set', type=str, default='eval2000',
                        help='decode set: [ eval2000 | rt02 | rt03 ]')

    # interpolation
    parser.add_argument('--interpolation_flag', type=int, default=0,
                        help='number of layers')
    parser.add_argument('--inter_path', type=str,
                        default="/project_bdda3/bdda/byxue/ami/s5c/exp/pytorch-Transformer-emb512_hid4096_nly6-ami+fisher-0.2-Bayesian-none-preFalse-no/model.pt",
                        #default="/project_bdda3/bdda/byxue/ami/s5c/exp/pytorch-LSTM-emb1024_hid1024_nly2-ami+fisher-0.2-Bayesian-0-preFalse-no/model.pt",
                        help="Path to a pretrained model for interpolation.")
    parser.add_argument('--inter_alpha', type=float, default=0.8,
                        help="Path to a pretrained model for interpolation.")

    args = parser.parse_args()
    print(args)
    assert os.path.exists(args.nbest_list), "Nbest list path does not exists."
    assert os.path.exists(args.vocabulary), "Vocabulary path does not exists."
    assert os.path.exists(args.model_path), "Model path does not exists."

    print(args.model_path)
    print("Load vocabulary", args.vocabulary)
    vocab = read_vocab(args.vocabulary)
    ntokens = len(vocab)
    print("Load model and criterion")

    import model
    if args.model == 'Transformer':
        # The activation function can be 'relu' (default) or 'gelu'
        if args.model_var == 'none':
            model_1 = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid,
                                           args.nlayers, 0.5, "gelu", True)
            print(model_1)
            if args.interpolation_flag != 0:
                model_2 = model.RNNModel('LSTM', ntokens, 1024, 1024, 2, 0.5, True)
                print(model_2)
                pass
            else:
                model_2 = None
                pass
            pass
        elif args.model_var == 'XL':
            model_1 = model.TransformerXLModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, 
                                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, tie_weight=True, tgt_len=args.seq_len, 
                                        ext_len=int(args.seq_len / 2), mem_len=int(args.seq_len / 2), 
                                        clamp_len=-1)
            if args.interpolation_flag != 0:
                model_2 = model.RNNModel('LSTM', ntokens, 1024, 1024, 2, 0.5, True)
                print(model_2)
                pass
            else:
                model_2 = None
                pass
            pass
        pass
    elif args.model == 'LSTM':
        if args.model_var == 'none' or args.model_var == 'back':
            model_1 = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                                   args.nlayers, 0.5, True)
            pass
        elif args.model_var == 'bid':
            model_1 = model.BiRNNModel(args.model, ntokens, args.emsize, args.nhid,
                                   args.nlayers, 0.5, True, True)
            pass
        print(model_1)

        if args.interpolation_flag != 0:
            model_2 = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                                    args.nlayers, 0.5, True)
            print(model_2)
        else:
            model_2 = None
        pass
    else:
        if args.model_var == 'none':
            model_1 = model.LSTMTransformerModel(ntokens, args.emsize, args.nhead, args.nhid,
                                           args.nlayers, args.emsize_l, args.nhid_l, 0.5, True)
            print(model_1)
            if args.interpolation_flag != 0:
                model_2 = model.TransformerModel(ntokens, args.emsize, args.nhead,
                                                      args.nhid, args.nlayers, 0.5, "gelu", True)
                print(model_2)
            else:
                model_2 = None
            pass
        pass
    pass

    args.inter_path = '/project_bdda6/bdda/byxue/swbd_wav/swbd/pytorch-LSTM-emb1024_hid1024_nly2-0.2-none-128-preFalse-base-1.7/model.pt'

    with open(args.model_path, 'rb') as f:
        model_dict_1 = torch.load(f, map_location=lambda storage, loc: storage)
    model1_dict = model_1.state_dict()
    model_dict_1 = {k: v for k, v in model_dict_1.items() if k in model1_dict}
    model1_dict.update(model_dict_1)
    model_1.load_state_dict(model1_dict)
        #print(n_model.state_dict())
        #if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        #    model.rnn.flatten_parameters()
    if args.interpolation_flag != 0:
        with open(args.inter_path, 'rb') as f:
            model_dict_2 = torch.load(f, map_location=lambda storage, loc: storage)
        model2_dict = model_2.state_dict()
        model_dict_2 =  {k: v for k, v in model_dict_2.items() if k in model2_dict}
        model2_dict.update(model_dict_2)
        model_2.load_state_dict(model2_dict)

    criterion = nn.CrossEntropyLoss()
    print("Load nbest list", args.nbest_list)
    nbest = load_nbest(args.nbest_list)
    # nbest = load_nbest('nbest_djj/text2')
    print("nbest length: ", len(nbest))
    #load nbest 
    #nbest = load_nbest(args.nbest_list)
    print("Compute sentence scores with a ", args.model, " model")
    model_1 = model_1.cuda()
    print("interpolation", args.interpolation_flag, args.inter_alpha)
    if args.interpolation_flag != 0:
        model_2 = model_2.cuda()
        print("model_2", model_2)

    nbest_and_scores = compute_scores(nbest, model_1, criterion, ntokens, vocab,
                                      model_type=args.model, inter_flag=args.interpolation_flag,
                                      alpha=args.inter_alpha, model_2=model_2, seg_wise=args.seg_wise, decode_set=args.decode_set, seq_len=args.seq_len, model_var=args.model_var)
    print("Write sentence scores out")
    write_scores(nbest_and_scores, args.outfile)


if __name__ == '__main__':
    main()
