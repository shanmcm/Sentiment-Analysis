import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.u = []
        self.linear = nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        self.W_attention = Parameter(torch.randn(self.hidden_dim * 2, self.hidden_dim * 2))
        self.bias_attention = Parameter(torch.randn(self.hidden_dim * 2))
        self.hidden_states = torch.Tensor()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, hidden_states):
        self.hidden_states = hidden_states
        # hidden_states = [batch_size, n_parole, embedding_dim]
        u_w = torch.randn(self.hidden_states.size()[1]).unsqueeze(0)
        s = torch.Tensor()
        for i in range(len(self.hidden_states)):
            u_it = torch.tanh(torch.mm(self.hidden_states[i].t(), self.W_attention.t()) + self.bias_attention)
            u_w = torch.mm(u_w, self.W_attention.t()) + self.bias_attention
            a = self.softmax(torch.mm(u_it, u_w.t()))
            s_i = torch.mm(self.hidden_states[i], a)
            s = torch.cat((s, s_i), dim=-1)
        return s