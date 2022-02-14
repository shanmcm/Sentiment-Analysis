import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import params


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = Parameter(torch.randn(4 * hidden_size, input_size))  # w_ih
        self.U = Parameter(torch.randn(4 * hidden_size, hidden_size))  # w_hh
        self.V = Parameter(torch.randn(4 * hidden_size, cell_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.cell_state = Parameter(torch.zeros(params.BATCH_SIZE, cell_size))

    def forward(self, input_data, state):  # state = (hx, cx)
        tmp1 = torch.mm(input_data, self.W.t())  # 32x3072 X  3072x4h -> 32x4h
        tmp2 = tmp1 + self.bias_ih  # -> 32x4h

        h_tmp = state[0]
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
        self.cell_state = Parameter(cy)
        return hy, cy

