# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import numpy as np
import six; from six.moves import cPickle as pkl

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nmt_model import AttNMT, translate_beam_k, translate_beam_1
from deeplib.text_data import TextPairIterator, TextIterator, read_dict
#import pdb

import re
from subprocess import Popen, PIPE

from deeplib.utils import timeSince, ids2words, unbpe

BOS_token = 0
EOS_token = 1
UNK_token = 2
# in the dict file, <s>/</s>=0, <unk>=1

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def train(model, optimizer, x_data, x_mask, y_data, y_mask, args):
    # Model.forward return loss
    loss = model(x_data, x_mask, y_data, y_mask)

    model.zero_grad()
    loss.sum().backward()
    # TODO: grad_clip
    optimizer.step()

    return loss.sum().item()

def train_model(args, model, train_iter, valid_iter):
    # data loading
    '''
    train_iter = TextPairIterator(args.train_src_file, args.train_trg_file,
                         args.src_dict, args.trg_dict, batch_size=args.batch_size,
                         maxlen=args.max_length,ahead=1000, resume_num=0)
    valid_iter = TextIterator(args.valid_src_file, args.src_dict,
                         batch_size=1, maxlen=1000,
                         ahead=1, resume_num=0)

    args.src_words_n = len(train_iter.src_dict2)
    args.trg_words_n = len(train_iter.trg_dict2)
    '''

    start = time.time()
    loss_total = 0  # Reset every args.print_every
    best_bleu = 0

    # model
    #model = AttNMT(args=args).to(device)
    #model.to(args.device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for x_data, x_mask, y_data, y_mask, cur_line, iloop in train_iter:

        loss = train(model, optimizer, x_data, x_mask, y_data, y_mask, args)
        loss_total += loss

        if iloop % args.print_every == 0:
            loss_avg = loss_total/args.print_every
            loss_total = 0
            print('%s: %d iters - %s %.4f' % (args.model_file, iloop, timeSince(start), loss_avg))

        if iloop >= 5000 and iloop % args.valid_every == 0:
            file_name = args.save_dir + '/' + args.model_file + '.pth'
            print ('saving the model to '+file_name)
            torch.save(model, file_name)

            print ('validating...')
            bleu_score = translate_file(args, valid=True, model=model)
            if os.path.exists(file_name+'.bleu'):
                with open(file_name+'.bleu', 'r') as bfp:
                    lines = bfp.readlines()
                    prev_bleus = [float(bs.split()[1]) for bs in lines]
            else:
                prev_bleus = [0]

            if bleu_score >= np.max(prev_bleus):
                best_bleu = bleu_score
                print('BEST BLEU : ', best_bleu)
                torch.save(model, file_name +'.best.pth')
                print('the best model is saved to '+file_name+'.best.pth')

            mode = 'w' if iloop == 10000 else 'a'
            with open(file_name+'.bleu', mode) as bfp:
                bfp.write(str(iloop) + '\t' + str(bleu_score) + ' ,\tBEST : ' + str(best_bleu) + ' ,\t Time : '+ timeSince(start) + '\n')
            print ('bleu_score', bleu_score)


def translate_file(args, valid=None, model=None):
    torch.no_grad()

    valid_iter = TextIterator(args.valid_src_file, args.src_dict,
                         batch_size=1, maxlen=1000,
                         ahead=1, resume_num=0)

    trg_dict2 = read_dict(args.trg_dict)

    args.trg_words_n = len(trg_dict2)

    trg_inv_dict = dict()
    for kk, vv in trg_dict2.items():
        trg_inv_dict[vv] = kk

    # model
    if model is None:

        file_name = args.save_dir + '/' + args.model_file + '.pth'
        if args.use_best == 1:
            file_name = file_name + '.best.pth'
            print("Using best model")
        model = torch.load(file_name)
    '''
    model = AttNMT(args=args)
    state_dict = tmp_model.module.state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    print("I'm using ", device)
    '''
    # translate
    if valid:
        multibleu_cmd = ["perl", args.bleu_script, args.valid_trg_file, "<"]
        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True)
    else:
        fp = open(args.trans_file, 'w')

    for x_data, x_mask, cur_line, iloop in valid_iter:
        if valid or args.beam_width == 1:
            samples = translate_beam_1(model, x_data, args)
        else:
            samples = translate_beam_k(model, x_data, args)
        sentence = ids2words(trg_inv_dict, samples, eos_id=EOS_token)
        sentence = unbpe(sentence)
        if valid:
            mb_subprocess.stdin.write(sentence + '\n')
            mb_subprocess.stdin.flush()
            if iloop % 500 == 0:
                print(iloop, 'is validated...')
        else:
            fp.write(sentence+'\n')
            if iloop % 500 == 0:
                print(iloop, 'is translated...')

    ret = -1
    if valid:
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        mb_subprocess.terminate()
        if out_parse:
            ret = float(out_parse.group()[6:])
    else:
        fp.close()

    torch.set_grad_enabled(True)
    return ret
