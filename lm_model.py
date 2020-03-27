from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import random
import copy

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from mylib.layers import CudaVariable, CudaVariableNoGrad, myEmbedding, myLinear, myLSTM, biLSTM

class LM(nn.Module):
    def __init__(self, args=None):
        super(LM, self).__init__()
        self.dim_enc = args.dim_enc
        self.dim_wemb = args.dim_wemb
        self.max_length = args.max_length
        self.rnn_name = args.rnn_name

        self.src_emb = myEmbedding(args.data_words_n, args.dim_wemb)

        if self.rnn_name == 'lstm':
            self.rnn_enc = nn.LSTM(args.dim_wemb, args.dim_enc, batch_first=False, bidirectional=False)
        elif self.rnn_name == 'mylstm':
            self.rnn_enc = myLSTM(args.dim_wemb, args.dim_enc, batch_first=False, direction='f')

        self.readout = myLinear(args.dim_enc, args.dim_wemb)
        self.dec = myLinear(args.dim_wemb, args.data_words_n)
        self.dec.weight = self.src_emb.weight # weight tying.

    def encoder(self, x_data, x_mask=None):
        Tx, Bn = x_data.size()
        x_emb = self.src_emb(x_data.view(Tx*Bn,1)) # Tx Bn
        x_emb = x_emb.view(Tx,Bn,-1)

        if x_mask is not None:
            x_emb = x_emb * x_mask.unsqueeze(2)

        # self.rnn_enc.flatten_parameters() # it removes a warning. but how? 
        if self.rnn_name == 'lstm':
            hidden, cell = self.init_hidden(Bn)
            output, hidden = self.rnn_enc(x_emb, (hidden, cell)) # Tx Bn E
        elif self.rnn_name == 'mylstm':
            output = self.rnn_enc(x_emb, x_mask=x_mask) # Tx Bn E

        if x_mask is not None:
            output = output * x_mask.unsqueeze(2)

        output = self.readout(output)

        #logit = torch.mm(output.view(Tx*Bn, -1), self.src_emb.weight.transpose(1,0))
        logit = self.dec(output)

        probs = F.log_softmax(logit, dim=2).view(Tx*Bn,-1)
        topv, yt = probs.topk(1)

        return probs.view(Tx, Bn, -1), yt.view(Tx, Bn)

    def init_hidden(self, Bn):
        hidden = CudaVariable(torch.zeros(1, Bn, self.dim_enc))
        cell = CudaVariable(torch.zeros(1, Bn, self.dim_enc))
        return hidden, cell
        
    def forward(self, data, mask=None):
        x_data = data[:-1]
        y_data = data[1:]
        x_data = CudaVariable(torch.LongTensor(x_data))
        y_data = CudaVariable(torch.LongTensor(y_data))

        if mask is None:
            x_mask = None
            y_mask = None
        else:
            x_mask = mask[1:]
            y_mask = mask[1:]
            x_mask = CudaVariable(torch.FloatTensor(x_mask))
            y_mask = CudaVariable(torch.FloatTensor(y_mask))

        Tx, Bn = x_data.size()
        x_emb = self.src_emb(x_data.view(Tx*Bn,1)) # Tx Bn
        x_emb = x_emb.view(Tx,Bn,-1)

        ht = CudaVariable(torch.zeros(Bn, self.dim_enc))
        ct = CudaVariable(torch.zeros(Bn, self.dim_enc))
        loss = 0
        criterion = nn.NLLLoss(reduce=False)
        for xi in range(Tx):
            ht, ct = self.rnn_enc.step(x_emb[xi,:,:], ht, ct, x_m=x_mask[xi])
            output = self.readout(ht)
            logit = self.dec(output)
            probs = F.log_softmax(logit, dim=1)
            #topv, yt = probs.topk(1)

            loss_t = criterion(probs, y_data[xi])
            if y_mask is not None:
                loss += torch.sum(loss_t*y_mask[xi])/Bn
            else:
                loss += torch.sum(loss_t)/Bn

        return loss

    def forward_old(self, data, mask=None):
        x_data = data[:-1]
        y_data = data[1:]
        x_data = CudaVariable(torch.LongTensor(x_data))
        y_data = CudaVariable(torch.LongTensor(y_data))

        if mask is None:
            x_mask = None
            y_mask = None
        else:
            x_mask = mask[1:]
            y_mask = mask[1:]
            x_mask = CudaVariable(torch.FloatTensor(x_mask))
            y_mask = CudaVariable(torch.FloatTensor(y_mask))

        probs, yt = self.encoder(x_data, x_mask)

        Ty, Bn = y_data.size()
        criterion = nn.NLLLoss(reduce=False)

        loss = criterion(probs.view(Ty*Bn,-1), y_data.view(Ty*Bn,))
        if y_mask is not None:
            loss = loss*y_mask.view(Ty*Bn,)
        loss = loss.sum()/Bn

        return loss
