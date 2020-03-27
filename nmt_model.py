from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import random
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from deeplib.layers import CudaVariable, myEmbedding, myLinear, myLSTM, biLSTM

BOS_token = 0
EOS_token = 1
UNK_token = 2
# in the dict file, <s>/</s>=0, <unk>=1

class AttNMT(nn.Module):
    def __init__(self, args=None):
        super(AttNMT, self).__init__()
        self.dim_enc = args.dim_enc
        self.dim_dec = args.dim_dec
        self.dim_wemb = args.dim_wemb
        self.dim_att = args.dim_att
        self.trg_words_n = args.trg_words_n # len(train_iter.trg_dict2)
        self.max_length = args.max_length
        self.bidir = True
        self.rnn_name = args.rnn_name

        #Embedding 부분
        self.src_emb = myEmbedding(args.src_words_n, args.dim_wemb) #input embedding
        self.dec_emb = myEmbedding(self.trg_words_n, self.dim_wemb) #output embedding


        if self.rnn_name == 'lstm':
            self.rnn_enc = nn.LSTM(args.dim_wemb, args.dim_enc, batch_first=False, bidirectional=self.bidir)
        elif self.rnn_name == 'bilstm':
            self.rnn_enc = biLSTM(args.dim_wemb, args.dim_enc, batch_first=False)
        elif self.rnn_name == 'mylstm':
            self.rnn_enc_f = myLSTM(args.dim_wemb, args.dim_enc, batch_first=False, direction='f')
            self.rnn_enc_r = myLSTM(args.dim_wemb, args.dim_enc, batch_first=False, direction='r')

        self.dec_h0 = myLinear(self.dim_enc*2, self.dim_dec) # dim_enc*2 because bidir
        self.dec_c0 = myLinear(self.dim_enc*2, self.dim_dec)

        self.att1 = myLinear(self.dim_enc*2 + self.dim_wemb + self.dim_dec, self.dim_att) # input??
        self.att2 = myLinear(self.dim_att, 1)

        self.rnn_step = nn.LSTMCell(self.dim_wemb + self.dim_enc*2, self.dim_dec) # is not dec?

        self.readout = myLinear(self.dim_enc*2 + self.dim_dec + self.dim_wemb, self.dim_wemb*2)
        # maxout between readout and logitout
        self.logitout = myLinear(self.dim_wemb, self.trg_words_n)

    def encoder(self, x_data, x_mask=None):
        Tx, Bn = x_data.size()
        # I can't understand this code ( how embedding is working?)
        x_emb = self.src_emb(x_data.view(Tx*Bn,1)) # TxBn E
        x_emb = x_emb.view(Tx,Bn,-1) # Tx Bn E

        if x_mask is not None:
            x_emb = x_emb * x_mask.unsqueeze(2)

        # self.rnn_enc.flatten_parameters() # it removes a warning. but how?
        if self.rnn_name == 'lstm':
            hidden, cell = self.init_hidden(Bn)
            output, hidden = self.rnn_enc(x_emb, (hidden, cell)) # Tx Bn E
        elif self.rnn_name == 'bilstm':
            output = self.rnn_enc(x_emb, x_mask) # Tx Bn E
        elif self.rnn_name == 'mylstm':
            rnn_f = self.rnn_enc_f(x_emb, x_mask=x_mask) # Tx Bn E
            rnn_r = self.rnn_enc_r(x_emb, x_mask=x_mask) # Tx Bn E
            output = torch.cat((rnn_f, rnn_r), dim=2)

        # rnn_enc receive [T, B, Emb_dim] data
        if x_mask is not None:
            output = output * x_mask.unsqueeze(2)
        return output

    def init_hidden(self, Bn):
        rnn_n = 2 if self.bidir else 1
        hidden = CudaVariable(torch.zeros(rnn_n, Bn, self.dim_enc))
        cell = CudaVariable(torch.zeros(rnn_n, Bn, self.dim_enc))
        return hidden, cell

    def dec_step(self, ctx, y_tm1, htm, ctm, xm=None):
        Tx, Bn, Ec = ctx.size()
        y_tm1 = y_tm1.view(Bn,) # Bn 1
        y_emb = self.dec_emb(y_tm1) # Bn E

        att_in = torch.cat((ctx, y_emb.expand(Tx, Bn, y_emb.size(1)) , htm.expand(Tx, Bn, htm.size(1))), dim=2)
        att1 = torch.tanh(self.att1(att_in)) # Tx Bn E
        att2 = self.att2(att1).view(Tx, Bn) # Tx Bn
        att2 = att2 - torch.max(att2)

        alpha = torch.exp(att2) if xm is None else torch.exp(att2)*xm
        alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
        ctx_t = torch.sum(alpha.unsqueeze(2) * ctx, dim=0) # Bn E

        dec_in = torch.cat((y_emb, ctx_t), dim=1)
        ht, ct = self.rnn_step(dec_in, (htm, ctm))

        readin = torch.cat((ctx_t, ht, y_emb), dim=1)
        readout = self.readout(readin)
        readout = readout.view(readout.size(0), self.dim_wemb, 2) # Bn Wemb 2
        read_max = torch.max(readout, dim=2)[0] # Bn Wemb

        logit = self.logitout(read_max)
        prob = F.log_softmax(logit, dim=1)
        topv, yt = prob.topk(1)

        return prob, yt.view(Bn,), ht, ct

    def forward(self, x_data, x_mask, y_data, y_mask):
        x_data = CudaVariable(torch.LongTensor(x_data))
        x_mask = CudaVariable(torch.FloatTensor(x_mask))
        y_data = CudaVariable(torch.LongTensor(y_data))
        y_mask = CudaVariable(torch.FloatTensor(y_mask))

        ctx = self.encoder(x_data, x_mask)

        Ty, Bn = y_data.size()

        ctx_sum = torch.sum(ctx*x_mask.unsqueeze(2), dim=0)
        ctx_mean = ctx_sum/torch.sum(x_mask, dim=0).unsqueeze(1)
        ht = torch.tanh(self.dec_h0(ctx_mean)) # h0
        ct = torch.tanh(self.dec_c0(ctx_mean)) # c0

        yt = CudaVariable(torch.zeros(Bn, )).type(torch.cuda.LongTensor) # BOS_token = 0

        loss = 0
        criterion = nn.NLLLoss(reduction='none')
        for yi in range(Ty):
            #print('ctx, yt, ht, ct', ctx.size(), yt.size(), ht.size(), ct.size())
            prob, yt, ht, ct = self.dec_step(ctx, yt, ht, ct, x_mask)
            loss_t = criterion(prob, y_data[yi])
            loss += torch.sum(loss_t * y_mask[yi])/Bn
            yt = y_data[yi]  # Teacher forcing

        return loss

def translate_encode(model, x_data, args):
    x_data = CudaVariable(torch.LongTensor(x_data))
    ctx = model.encoder(x_data)
    ctx_mean = torch.mean(ctx, dim=0)
    ht = torch.tanh(model.dec_h0(ctx_mean)) # h0
    ct = torch.tanh(model.dec_c0(ctx_mean)) # h0
    yt = CudaVariable(torch.zeros(1, )).type(torch.cuda.LongTensor) # y0, BOS_token=0

    return ctx, yt, ht, ct

def translate_beam_1(model, x_data, args):
    ctx, yt, ht, ct = translate_encode(model, x_data, args)

    y_hat = []
    for yi in range(args.max_length):
        prob, yt, ht, ct = model.dec_step(ctx, yt, ht, ct)
        y_hat.append(yt)
        if yt[0] == EOS_token:
            break

    y_hat = torch.stack(y_hat)
    y_hat = y_hat.cpu().numpy().flatten().tolist()

    return y_hat

def translate_beam_k(model, x_data, args):
    ctx, yt, ht, ct = translate_encode(model, x_data, args)

    sample_sent = []
    sample_score = []

    k = args.beam_width
    live_k = 1
    dead_k = 0

    hyp_samples = [[]]
    hyp_scores = CudaVariable(torch.zeros(live_k,))
    hyp_states_h = []
    hyp_states_c = []

    for yi in range(args.max_length):
        ctx_k = ctx.expand(ctx.size(0), live_k, ctx.size(2))
        pt, yt, ht, ct = model.dec_step(ctx_k, yt, ht, ct)

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
            if new_hyp_samples[idx][-1].cpu().numpy() == EOS_token: # EOS
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
