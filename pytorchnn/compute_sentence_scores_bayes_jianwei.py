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


def load_swbd(path):
    nbest = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            nbest.append(line)
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
    return word2idx


def get_input_and_target(hyp, vocab):
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
    #input_string = hyp
    #output_string = hyp
    input_ids, output_ids = [], []
    for word in input_string.split():
        try:
            input_ids.append(vocab[word]) # word.lower()
        except KeyError:
            input_ids.append(vocab['<unk>'])
    for word in output_string.split():
        try:
            output_ids.append(vocab[word]) # word.lower()
        except KeyError:
            output_ids.append(vocab['<unk>'])
    #return input_ids[0:-1], output_ids[1:]
    print("input_ids", input_ids)
    return input_ids, output_ids


def compute_sentence_score(model, criterion, ntokens, data, target,
                           model_type='LSTM', hidden_1=None, inter_flag=False, alpha=0.0, model_2=None, hidden_2=None):
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

    length = len(data)
    #print(data.size())
    data = torch.LongTensor(data).view(-1, 1).contiguous().cuda()
    target = torch.LongTensor(target).view(-1).contiguous().cuda()
    with torch.no_grad():
        if model_type == 'Transformer':
            output_1 = model(data)
        else:
            output_1, hidden_1 = model(data, hidden_1)

        # XBY 10.4, interpolation
        if inter_flag == 1:
            if model_type == 'Transformer':
                output_2 = model_2(data)
            else:
                output_2, hidden_2 = model_2(data, hidden_2)
            #print("alpha", alpha)
            output = alpha * output_1 + (1. - alpha) * output_2
        else:
            output = output_1
            hidden_2 = None

        loss = criterion(output.view(-1, ntokens), target)
    print(loss)
    sent_score = (length) * loss.item()
    if model_type == 'Transformer':
        return sent_score
    return sent_score, hidden_1, hidden_2


def compute_scores_swbd(nbest, model, criterion, ntokens, vocab, model_type='LSTM', inter_flag=False, alpha=0.0, model_2=None):
    model.eval()
    if inter_flag == 1:
        model_2.eval()
    total_score = 0
    total_len = 0
    nbest_and_scores = defaultdict(float)
    if model_type != 'Transformer':
        hidden_1 = model.init_hidden(1)
        if inter_flag == 1:
            hidden_2 = model_2.init_hidden(1)
        else:
            hidden_2 = None

    for key in nbest:
        x, target = get_input_and_target(key, vocab)
        if model_type == 'Transformer':
            score = compute_sentence_score(model, criterion, ntokens, x,
                                           target, model_type, inter_flag=inter_flag, alpha=alpha, model_2=model_2)
        else:
            score, hidden_1, hidden_2 = compute_sentence_score(model, criterion, ntokens, x,
                                           target, model_type, hidden_1, inter_flag=inter_flag, alpha=alpha, model_2=model_2, hidden_2=hidden_2)
        total_len += len(x)
        total_score += score

    print(total_len)
    print(math.exp(total_score / total_len))
    return nbest_and_scores


def compute_scores(nbest, model, criterion, ntokens, vocab, model_type='LSTM', inter_flag=False, alpha=0.0, model_2=None):
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
    nbest_and_scores = defaultdict(float)
    if model_type != 'Transformer':
        hidden_1 = model.init_hidden(1)
        if inter_flag == 1:
            hidden_2 = model_2.init_hidden(1)
        else:
            hidden_2 = None
    for key in nbest.keys():
        #print("key:", key)
        if model_type != 'Transformer':
            cached_hiddens_1 = []
            cached_hiddens_2 = []
        for hyp in nbest[key]:
            print("hyp:", hyp)
            x, target = get_input_and_target(hyp, vocab)
            if model_type == 'Transformer':
                score = compute_sentence_score(model, criterion, ntokens, x,
                                               target, model_type, inter_flag=inter_flag, alpha=alpha, model_2=model_2)
            else:
                score, new_hidden_1, new_hidden_2 = compute_sentence_score(model, criterion,
                                                           ntokens, x, target,
                                                           model_type, hidden_1, inter_flag, alpha=alpha, model_2=model_2, hidden_2=hidden_2)
                cached_hiddens_1.append(new_hidden_1)
                if inter_flag == 1:
                    cached_hiddens_2.append(new_hidden_2)    
            total_len += len(x)
            total_score += score
            if key in nbest_and_scores:
                nbest_and_scores[key].append((hyp, score))
            else:
                nbest_and_scores[key] = [(hyp, score)]
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
            hidden_1 = cached_hiddens_1[0]
            if inter_flag == 1:
                hidden_2 = cached_hiddens_2[0]

    print(total_len)
    print(total_score / total_len)
    #print(nbest_and_scores)

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
    # XBY 2.19: uncertainty type selection
    parser.add_argument('--uncertainty', type=str, default='none',
                        help='uncertainty type: [none | Bayesian | Gaussian | Varationa | Dirichlet]')
    parser.add_argument('--T_bayes_pos', type=str, default='none',
                        help='Transformer Bayesian type: [None | FFN | MHA | EMB]')
    parser.add_argument('--L_bayes_pos', type=int, default=0,
                        help='LSTM Bayesian position: [None | 1: input_gate | 2: forget_gate | 3: cell_gate | 4: output_gate]')
    parser.add_argument('--L_gauss_pos', type=int, default=0,
                        help='LSTM Gaussian position: [0: None | 1: input_gate | 2: forget_gate | 3: cell_gate | 4: output_gate'
                             ' | 5: cell | 6: hidden | 7: inputs]')
    parser.add_argument('--T_gauss_pos', type=int, default=0,
                        help='Transformer Gaussian position: [0: None | 1: FFN]')

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
        if args.uncertainty == 'none':
            model_1 = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid,
                                           args.nlayers, 0.5, "gelu", args.tied)
        elif args.uncertainty == 'Bayesian':
            model_1 = model.BayesTransformerModel(ntokens, args.emsize, args.nhead, args.nhid,
                                                args.nlayers, 0.5, True, args.T_bayes_pos)
            if args.interpolation_flag == 1:
                model_2 = model.BayesTransformerModel(ntokens, args.emsize, args.nhead,
                                                      args.nhid, args.nlayers, 0.5, True, 'none')
                print(model_2)
            else:
                model_2 = None
        elif args.uncertainty == 'Gaussian':
            model_1 = model.GaussTransformerModel(ntokens, args.emsize, args.nhead, args.nhid,
                                                args.nlayers, 0.5, True, args.T_gauss_pos)
            if args.interpolation_flag == 1:
                model_2 = model.GaussTransformerModel(ntokens, args.emsize, args.nhead,
                                                      args.nhid, args.nlayers, 0.5, True, 0)
                print(model_2)
            else:
                model_2 = None
    else:
        if args.uncertainty == 'none':
            model_1 = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                                   args.nlayers, 0.5, args.tied)
            print(model_1)
        elif args.uncertainty == 'Bayesian':
            model_1 = model.BayesRNNModel(args.model, ntokens, args.emsize, args.nhid,
                                        args.nlayers, 0.5, args.tied, args.L_bayes_pos)
            print(model_1)
            if args.interpolation_flag == 1:
                model_2 = model.BayesRNNModel(args.model, ntokens, args.emsize, args.nhid,
                                              args.nlayers, 0.5, False, 0)
                print(model_2)
            else:
                model_2 = None
        elif args.uncertainty == 'Gaussian':
            model_1 = model.GaussRNNModel(args.model, ntokens, args.emsize, args.nhid,
                                        args.nlayers, 0.5, False, args.L_gauss_pos)
            print(model_1)
            if args.interpolation_flag == 1:
                model_2 = model.GaussianRNNModel(args.model, ntokens, args.emsize, args.nhid,
                                              args.nlayers, 0.5, False, 0)
                print(model_2)
            else:
                model_2 = None
            pass
        pass
    pass

    with open(args.model_path, 'rb') as f:
        model_dict = torch.load(f, map_location=lambda storage, loc: storage)
        model_1.load_state_dict(model_dict)
        #print(n_model.state_dict())
        #if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        #    model.rnn.flatten_parameters()
    if args.interpolation_flag == 1:
        with open(args.inter_path, 'rb') as f:
            model_dict_2 = torch.load(f, map_location=lambda storage, loc: storage)
            model_2.load_state_dict(model_dict_2)
            #print(n_model_2.state_dict())

    criterion = nn.CrossEntropyLoss()
    print("Load nbest list", args.nbest_list)
    nbest = load_nbest(args.nbest_list)
    #nbest = load_swbd('/project_bdda3/bdda/byxue/s5c/s-swbd-test.txt')
    print("nbest length: ", len(nbest))
    #load nbest 
    #nbest = load_nbest(args.nbest_list)
    print("Compute sentence scores with a ", args.model, " model")
    model_1 = model_1.cuda()
    print("interpolation", args.interpolation_flag, args.inter_alpha)
    if args.interpolation_flag == 1:
        model_2 = model_2.cuda()
        print("model_2", model_2)

    nbest_and_scores = compute_scores(nbest, model_1, criterion, ntokens, vocab,
                                      model_type=args.model, inter_flag=args.interpolation_flag,
                                      alpha=args.inter_alpha, model_2=model_2)
    print("Write sentence scores out")
    write_scores(nbest_and_scores, args.outfile)


if __name__ == '__main__':
    main()
