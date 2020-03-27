from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import random
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from layers import myEmbedding, myLinear, myLSTM

use_cuda = torch.cuda.is_available()
UNK_token = 2
EOS_token = 1
BOS_token = 0

def CudaVariable(X):
    return Variable(X).cuda() if use_cuda else Variable(X)

class AttNMT(nn.Module):
    def __init__(self, args=None):
        super(AttNMT, self).__init__()
        self.dim_enc = args.dim_enc
        self.dim_dec = args.dim_dec
        self.dim_wemb = args.dim_wemb
        self.dim_att = args.dim_att
        self.trg_words_n = args.trg_words_n
        self.max_length = args.max_length
        self.bidir = True
        self.rnn_name = args.rnn_name

        self.src_emb = myEmbedding(args.src_words_n, args.dim_wemb)
        self.dec_emb = myEmbedding(self.trg_words_n, self.dim_wemb)

        #self.rnn_enc= nn.LSTM(args.dim_wemb, args.dim_enc, batch_first=False, bidirectional=self.bidir)
        self.rnn_enc_f = myLSTM(args.dim_wemb, args.dim_enc, batch_first=False)
        self.rnn_enc_r = myLSTM(args.dim_wemb, args.dim_enc, direction='r', batch_first=False)

        self.dec_h0 = myLinear(self.dim_enc*2, self.dim_dec)
        self.dec_c0 = myLinear(self.dim_enc*2, self.dim_dec)

        self.att1 = myLinear(self.dim_enc*2 + self.dim_wemb + self.dim_dec, self.dim_att)
        self.att2 = myLinear(self.dim_att, 1)

        self.rnn_step = nn.LSTMCell(self.dim_wemb + self.dim_enc*2, self.dim_dec)

        self.readout = myLinear(self.dim_enc*2 + self.dim_dec + self.dim_wemb, self.dim_wemb*2)
        # maxout between readout and logitout
        self.logitout = myLinear(self.dim_wemb, self.trg_words_n)

    def encoder(self, x_data, x_mask=None):
        Tx, Bn = x_data.size()
        
        x_emb = self.src_emb(x_data.view(Tx*Bn,1)) # Tx Bn
        x_emb = x_emb.view(Tx,Bn,-1)
        
        #hidden, cell = self.init_hidden(Bn)
        #output, hidden = self.rnn_enc(x_emb, (hidden, cell)) # Tx Bn E
        rnn_f = self.rnn_enc_f(x_emb, x_mask=x_mask) 
        if x_mask is None:
            rnn_r = self.rnn_enc_r(x_emb) 
        else:
            rnn_r = self.rnn_enc_r(x_emb, x_mask=x_mask) 
        
        output = torch.cat((rnn_f, rnn_r), dim=2)

        if x_mask is not None:
            output = output * x_mask.unsqueeze(2)
        return output

    def init_hidden(self, Bn):
        rnn_n = 2 if self.bidir else 1
        if self.rnn_name == 'gru':
            return CudaVariable(torch.zeros(rnn_n, Bn, self.dim_enc))
        elif self.rnn_name == 'lstm':
            hidden = CudaVariable(torch.zeros(rnn_n, Bn, self.dim_enc))
            cell = CudaVariable(torch.zeros(rnn_n, Bn, self.dim_enc))
            return hidden, cell
        
    def dec_step(self, ctx, y_tm1, htm, ctm, xm=None, ym=None):
        Tx, Bn, Ec = ctx.size()
        y_tm1 = y_tm1.view(Bn,) # Bn 1
        y_emb = self.dec_emb(y_tm1) 

        att_in = torch.cat((ctx, y_emb.expand(Tx, Bn, y_emb.size()[1]) , htm.expand(Tx, Bn, htm.size()[1])), dim=2)
        att1 = F.tanh(self.att1(att_in)) # Tx Bn E 
        att2 = self.att2(att1).view(Tx, Bn) # Tx Bn 
        att2 = att2 - torch.max(att2) 

        alpha = torch.exp(att2) if xm is None else torch.exp(att2)*xm
        alpha = alpha / (torch.sum(alpha, dim=0, keepdim=True) + 1e-15)
        ctx_t = torch.sum(alpha.unsqueeze(2) * ctx, dim=0) # Bn E 

        dec_in = torch.cat((y_emb, ctx_t), dim=1)
        if self.rnn_name== 'gru':
            ht, ct = self.rnn_step(dec_in, htm), None
        elif self.rnn_name == 'lstm':
            ht, ct = self.rnn_step(dec_in, (htm, ctm))

        #if ym is not None:
        #    ym_neg = 1.0 - ym
        #    ht = ym.unsqueeze(1)*ht + ym_neg.unsqueeze(1)*htm
        #    if self.rnn_name == 'lstm':
        #        ct = ym.unsqueeze(1)*ct + ym_neg.unsqueeze(1)*ctm

        readin = torch.cat((ctx_t, ht, y_emb), dim=1)
        readout = self.readout(readin)
        readout = readout.view(readout.size()[0], self.dim_wemb, 2) # Bn Wemb 2
        read_max = torch.max(readout, dim=2)[0] # Bn Wemb

        logit = self.logitout(read_max)
        prob = F.log_softmax(logit, dim=1)
        topv, yt = prob.data.topk(1)

        return prob, yt.view(Bn,), ht, ct

    def forward(self, x_data, x_mask, y_data, y_mask):
        ctx = self.encoder(x_data, x_mask)
        
        Tx, Bn, E = ctx.size()
        Ty, Bn = y_data.size()

        ctx_sum = torch.sum(ctx*x_mask.unsqueeze(2), dim=0)
        ctx_mean = ctx_sum/torch.sum(x_mask, dim=0).unsqueeze(1)
        ht = F.tanh(self.dec_h0(ctx_mean)) # h0 
        if self.rnn_name == 'gru':
            ct = None
        elif self.rnn_name == 'lstm':
            ct = F.tanh(self.dec_c0(ctx_mean)) # h0 
        
        yt = CudaVariable(torch.zeros(Bn, )).type(torch.cuda.LongTensor) # BOS_token = 0

        loss = 0
        criterion = nn.NLLLoss(reduce=False)
        for yi in range(Ty):
            prob, yt, ht, ct = self.dec_step(ctx, yt, ht, ct, x_mask, y_mask[yi])
            loss_t = criterion(prob, y_data[yi])
            loss += torch.sum(loss_t * y_mask[yi])/Bn
            yt = y_data[yi]  # Teacher forcing

        return loss

def translate(model, x_data, args):
    
    ctx = model.encoder(x_data)
    ctx_mean = torch.mean(ctx, dim=0)
    ht = F.tanh(model.dec_h0(ctx_mean)) # h0 
    if args.rnn_name == 'gru':
        ct = None
    elif args.rnn_name == 'lstm':
        ct = F.tanh(model.dec_c0(ctx_mean)) # h0 
    yt = Variable(torch.zeros(1, ), volatile=True) # y0, BOS_token=0

    y_data = []
    for yi in range(args.max_length):
        yt = yt.cuda().type(torch.cuda.LongTensor) 
        prob, yt, ht, ct = model.dec_step(ctx, yt, ht, ct)
        y_data.append(yt[0])
        if yt[0] == EOS_token:
            break
        yt = Variable(torch.ones(1,)*yt[0], volatile=True) # Free running
        
    return y_data

