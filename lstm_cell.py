import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import params


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size): # in teoria hidden_size e cell_size uguali ma per adesso passiamo entrambi
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = Parameter(torch.randn(4 * hidden_size, input_size)) #w_ih
        self.U = Parameter(torch.randn(4 * hidden_size, hidden_size)) #w_hh
        self.V = Parameter(torch.randn(4 * hidden_size,  cell_size)) #controllare se è giusto hidden size
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.cell_state = Parameter(torch.zeros(params.BATCH_SIZE, cell_size))
        self.W_attention = Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_attention = Parameter(torch.randn(hidden_size))
        self.hidden_states = [Parameter(torch.zeros(params.BATCH_SIZE, hidden_size))]

    def forward(self, input, state): #state = (hx, cx)
        tmp1 = torch.mm(input, self.W.t())  # 32x3072 X  3072x4h -> 32x4h
        tmp2 = tmp1 + self.bias_ih  # -> 32x4h

        h_tmp = self.hidden_states[-1]
        tmp3 = torch.mm(h_tmp, self.U.t())

        c_tmp = self.cell_state
        tmp4 = torch.mm(c_tmp, self.V.t())
        gates = (tmp2 + tmp3 + tmp4)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy_1 = (forgetgate * state[1])
        cy_2 = (ingate * cellgate)
        cy = cy_1 + cy_2
        hy = outgate * torch.tanh(cy)
        self.hidden_states = self.hidden_states + [hy]
        self.cell_state = Parameter(cy)
        return hy, cy

    def attention_layer(self):
      u = []
      for i in range(len(self.hidden_states)):
        u_it = torch.tanh(torch.mm(self.W_attention, self.hidden_states[i])+self.bias_attention) # il nostro paper usa tan ma abbiamo messo tanh
        u.append(u_it)
      attention_vector = torch.randn(self.hidden_size) #u_w global context vector
      a = torch.softmax(torch.mm(u, attention_vector))
      s = torch.mm(a, self.hidden_state)
