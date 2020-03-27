import torch 
import math
from torch.nn.parameter import Parameter

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def CudaVariable(X):
    return Variable(X).to(device)

def get_scale(nin, nout):
    return  1. / math.sqrt(6)/math.sqrt(nin+nout) # Xavier

class myEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(myEmbedding, self).__init__(num_embeddings, embedding_dim)

    def reset_parameters(self):
        scale = get_scale(self.num_embeddings, self.embedding_dim)
        self.weight.data.uniform_(-scale, scale)

class myLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(myLinear, self).__init__(in_features, out_features)

    def reset_parameters(self):
        #scale = 1. / math.sqrt(self.weight.size(1))
        scale = get_scale(self.in_features, self.out_features)
        self.weight.data.uniform_(-scale, scale)
        if self.bias is not None:
            self.bias.data.uniform_(-scale, scale)

class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, direction='f', batch_first=False):
        """Initialize params."""
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.direction = direction

        self.input_weights = myLinear(input_size, 4*hidden_size)
        self.hidden_weights = myLinear(hidden_size, 4*hidden_size)

    def step(self, xt, htm, ctm, x_m=None):
        gates = self.input_weights(xt) + self.hidden_weights(htm)
        ig, fg, og, ct = gates.chunk(4, 1)

        ig = F.sigmoid(ig)
        fg = F.sigmoid(fg)
        og = F.sigmoid(og)
        ct = F.tanh(ct)  # o_t

        #print('fg', fg.size(), 'ctm', ctm.size(), 'ig', ig.size(), 'ct', ct.size())
        cy = (fg * ctm) + (ig * ct)
        hy = og * F.tanh(cy)
        if x_m is not None:
            cy = cy * x_m[:,None] + ctm*(1. - x_m)[:,None]
            cy = hy * x_m[:,None] + htm*(1. - x_m)[:,None]

        return hy, cy

    def forward(self, X, x_mask=None, hidden=None):
        if self.batch_first:
            X = X.transpose(0, 1)

        if hidden is None:
            ht, ct = self.init_hidden(X.size(1))

        if self.direction == 'f':
            steps = range(0, X.size(0)) # 'forward'
        else:
            steps = range(X.size(0)-1,-1,-1) # 'backward' or 'reverse'

        output = []
        for i in steps:
            if x_mask is None:
                ht, ct = self.step(X[i], ht, ct)
            else:
                ht, ct = self.step(X[i], ht, ct, x_m=x_mask[i])
            if self.direction == 'f':
                output.append(ht)
            else:
                output.insert(0, ht)
        output = torch.cat(output, 0).view(X.size(0), *output[0].size()) # list 2 tensor

        if self.batch_first:
            output = output.transpose(0, 1)

        return output

    def init_hidden(self, Bn):
        hidden = CudaVariable(torch.zeros(Bn, self.hidden_size))
        cell = CudaVariable(torch.zeros(Bn, self.hidden_size))
        return hidden, cell
