import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import normalize
import params


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.u = []
        self.linear = nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        self.W_attention = Parameter(torch.randn(self.hidden_dim * 2, self.hidden_dim * 2, device=params.DEVICE))
        self.bias_attention = Parameter(torch.randn(self.hidden_dim * 2, device=params.DEVICE))
        self.hidden_states = torch.tensor((), device=params.DEVICE)
        self.softmax = nn.Softmax(dim=0)  # , eps=1e-5)

    def forward(self, hidden_states):
        self.hidden_states = hidden_states
        self.hidden_states.to(params.DEVICE)
        u_w = torch.randn(self.hidden_states.size()[1], device=params.DEVICE).unsqueeze(0)
        s = torch.tensor((), device=params.DEVICE)
        attention = torch.tensor((), device=params.DEVICE)
        for i in range(len(self.hidden_states)):
            u_it = torch.tanh(torch.mm(self.hidden_states[i].t(), self.W_attention.t()) + self.bias_attention)
            u_w = torch.mm(u_w, self.W_attention.t()) + self.bias_attention
            u_w = normalize(u_w)
            a = self.softmax(torch.mm(u_it, u_w.t()))
            attention = torch.cat((attention, a), dim=1)
            s_i = torch.mm(self.hidden_states[i], a)
            s_i = torch.transpose(s_i, 0, 1)
            s = torch.cat((s, s_i), dim=0)
            s.to(params.DEVICE)
        return s, attention
