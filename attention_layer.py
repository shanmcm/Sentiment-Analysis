import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.u = []
        self.W_attention = Parameter(torch.randn(self.hidden_dim*2, self.hidden_dim*2))
        self.bias_attention = Parameter(torch.randn(self.hidden_dim*2))
        self.hidden_states = []
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, hidden_states):
        self.hidden_states = hidden_states
        for i in range(len(self.hidden_states)):
            u_it = torch.tanh(torch.mm(self.hidden_states[i], self.W_attention.t()) + self.bias_attention)
        self.u.append(u_it)
        attention_vector = torch.randn(self.hidden_dim*2)  # u_w global context vector
        tmp = torch.mm(attention_vector, torch.stack(self.u).t())
        a = self.softmax(tmp)
        s = torch.mm(a, self.hidden_states)
        # mi aspetto un output di 32x256
        return s
