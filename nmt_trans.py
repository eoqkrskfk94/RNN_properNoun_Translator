# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import random
import copy
import os
from io import open
import time
import re
from subprocess import Popen, PIPE

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from mylib.text_data import TextIterator, read_dict
from mylib.utils import timeSince, ids2words, unbpe
from mylib.layers import CudaVariable

import nmt_const as Const
from Beam import Beam

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def translate_attall_old(model, x_data, x_mask, args):
    x_data = CudaVariable(torch.LongTensor(x_data)) # T B
    x_mask = CudaVariable(torch.LongTensor(x_mask)) # T B

    x_data = x_data.transpose(0, 1) # B T
    x_mask = x_mask.transpose(0, 1) # B T

    Bn, Tx = x_data.size()
    encY = model.encoder(x_data, x_mask)
    
    y_hat = CudaVariable(torch.ones(Bn)*Const.BOS).type(torch.cuda.LongTensor)
    for yi in range(args.max_length+2):
        len_dec_seq = yi + 1
        dec_seq = y_hat.view(Bn, -1)

        dec_pos = torch.arange(1, len_dec_seq + 1).view(Bn, -1)
        dec_pos = CudaVariable(dec_pos).type(torch.cuda.LongTensor)

        decY = model.decoder(dec_seq, dec_pos, x_data, encY)
        decY = model.trg_word_proj(decY)
        prob = model.prob_proj(decY.view(-1, decY.size(2)))
        topv, yt = prob.topk(1)

        y_hat = torch.cat((dec_seq, yt[-1,0].view(1,1)), dim=1)
        if yi > 0 and yt[-1,0] == Const.EOS:
            break

    y_hat = y_hat.cpu().numpy().flatten().tolist()

    return y_hat

def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, Bs):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * Bs, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor

def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list, Bs):
    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.cuda.LongTensor(active_inst_idx, device=device)

    active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, Bs)
    active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, Bs)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return active_src_seq, active_src_enc, active_inst_idx_to_position_map

def beam_decode_step(model, inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, Bs):
    ''' Decode and update beam status, and then return active beam idx '''

    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def prepare_beam_dec_pos(len_dec_seq, n_active_inst, Bs):
        dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=device)
        dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * Bs, 1)
        return dec_partial_pos

    def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, Bs):
        dec_output = model.decoder(dec_seq, dec_pos, src_seq, enc_output)
        dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
        word_prob = F.log_softmax(model.trg_word_proj(dec_output), dim=1)
        word_prob = word_prob.view(n_active_inst, Bs, -1)

        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)

    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, Bs)
    word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, Bs)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(
        inst_dec_beams, word_prob, inst_idx_to_position_map)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores


def translate_attall(model, x_data, x_mask, args):
    x_data = CudaVariable(torch.LongTensor(x_data)) # T B
    x_mask = CudaVariable(torch.LongTensor(x_mask)) # T B

    x_data = x_data.transpose(0, 1) # B T
    x_mask = x_mask.transpose(0, 1) # B T
    xm = (x_data.data.ne(Const.PAD)).type(torch.cuda.FloatTensor)

    Bs = args.beam_width
    Bn, Tx = x_data.size()
    encY = model.encoder(x_data, x_mask) * xm.unsqueeze(2)
    
    #-- Repeat data for beam search
    n_inst, Ts, d_h = encY.size()
    x_data = x_data.repeat(1, Bs).view(n_inst * Bs, Ts)
    encY = encY.repeat(1, Bs, 1).view(n_inst * Bs, Ts, d_h)

    #-- Prepare beams
    inst_dec_beams = [Beam(Bs, device=device) for _ in range(n_inst)]

    #-- Bookkeeping for active or not
    active_inst_idx_list = list(range(n_inst))
    inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    #-- Decode
    for len_dec_seq in range(1, args.max_length + 1):
        active_inst_idx_list = beam_decode_step(model, inst_dec_beams, len_dec_seq, x_data, encY, inst_idx_to_position_map, Bs)

        if not active_inst_idx_list:
            break  # all instances have finished their path to <EOS>

        x_data, encY, inst_idx_to_position_map = collate_active_info(
            x_data, encY, inst_idx_to_position_map, active_inst_idx_list, Bs)

    n_best = 1
    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)

    y_hat = batch_hyp[0][0]#cpu().numpy().flatten().tolist()
    return y_hat

def translate_nmt(model, x_data, args):
    ctx, yt, ht, ct = model.translate_encode(x_data)

    pos_idx = torch.from_numpy(np.arange(args.max_length+2)).type(torch.cuda.LongTensor)
    if args.beam_width == 1:
        y_hat = []
        for yi in range(args.max_length+2):
            prob, yt, ht, ct = model.dec_step(pos_idx[yi], ctx, yt, ht, ct)
            y_hat.append(yt)
            if yt[0] == Const.EOS:
                break
        y_hat = torch.stack(y_hat)
        y_hat = y_hat.cpu().numpy().flatten().tolist()

        return y_hat

    # for beam size > 1
    sample_sent = []
    sample_score = []

    k = args.beam_width
    live_k = 1 
    dead_k = 0 

    hyp_samples = [[]]
    hyp_scores = CudaVariable(torch.zeros(live_k,))
    hyp_states_h = []
    hyp_states_c = []
    
    for yi in range(args.max_length+2):
        ctx_k = ctx.expand(ctx.size(0), live_k, ctx.size(2))
        pt, yt, ht, ct = model.dec_step(pos_idx[yi], ctx_k, yt, ht, ct)

        cand_scores = hyp_scores.unsqueeze(1).expand(live_k, pt.size(1)) - pt
        cand_flat = cand_scores.view(-1)
        values, ranks_flat = torch.sort(cand_flat)
        ranks_flat = ranks_flat[:(k-dead_k)]

        voc_size = pt.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = Variable(torch.zeros(k-dead_k)).cuda()
        new_hyp_states_h = []
        new_hyp_states_c = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            ti = int(ti)
            new_hyp_samples.append(hyp_samples[ti]+[wi])
            new_hyp_scores[idx] = costs[idx]
            new_hyp_states_h.append(ht[ti])
            new_hyp_states_c.append(ct[ti])

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states_h = []
        hyp_states_c = []

        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1].cpu().numpy() == Const.EOS: # EOS
                sample_sent.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states_h.append(new_hyp_states_h[idx])
                hyp_states_c.append(new_hyp_states_c[idx])
        
        live_k = new_live_k
        if new_live_k > 0:
            hyp_scores = torch.stack(hyp_scores)
        else:
            break
        if dead_k >= k:
            break

        yt = torch.stack([w[-1] for w in hyp_samples])
        ht = torch.stack(hyp_states_h)
        ct = torch.stack(hyp_states_c)

    # dump every remaining one
    if live_k > 0:
        for idx in range(live_k):
            sample_sent.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    # length normalization
    scores = [score/len(sample) for (score, sample) in zip(sample_score, sample_sent)]
    scores = torch.stack(scores).cpu().detach().numpy()
    best_sample = sample_sent[scores.argmin()]

    y_hat = torch.stack(best_sample)
    y_hat = y_hat.cpu().numpy().flatten().tolist()

    return y_hat

def translate_file(model, args, valid=None):
    model.eval()
    torch.no_grad()
    
    mask_pos = True if args.model == 'attall' else False
    valid_iter = TextIterator(args.valid_src_file, args.src_dict, 
                    batch_size=1, maxlen=1000, ahead=1, resume_num=0, 
                    mask_pos=mask_pos, const_id=Const)
    trg_dict2 = read_dict(args.trg_dict, const_id=Const)
    args.trg_words_n = len(trg_dict2)

    trg_inv_dict = dict()
    for kk, vv in trg_dict2.items():
        trg_inv_dict[vv] = kk

    # translate
    if valid:
        multibleu_cmd = ["perl", args.bleu_script, args.valid_trg_file, "<"]
        mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE, 
                                universal_newlines=True, encoding='utf-8')
    else:
        fp = open(args.trans_file, 'w')

    for x_data, x_mask, cur_line, iloop in valid_iter:

        if args.model == 'attall':
            samples = translate_attall(model, x_data, x_mask, args)
        else:
            samples = translate_nmt(model, x_data, args)

        sentence = ids2words(trg_inv_dict, samples, eos_id=Const.EOS)
        sentence = unbpe(sentence)
        if valid: 
            mb_subprocess.stdin.write(sentence + '\n')
            mb_subprocess.stdin.flush()
        else:
            #print(samples)
            #print(sentence)

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

