import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.u = []
        self.W_attention = Parameter(torch.randn(self.hidden_dim * 2, self.hidden_dim * 2))
        self.bias_attention = Parameter(torch.randn(self.hidden_dim * 2))
        self.hidden_states = torch.Tensor()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, hidden_states):
        self.hidden_states = hidden_states

        # Note: torch.mv(m,v) is product between matrix (m) and vector (v)
        # Define u (222, 32, 256)
        for i in range(len(self.hidden_states)):
            u_it = torch.tanh(torch.mm(self.hidden_states[i], self.W_attention.t()) + self.bias_attention)
            # Questo self.u.append che segue prima non era indented ma credo fosse sbagliato visto che altrimenti facciamo append solo sull'ultimo
            self.u.append(u_it)
        tmp = torch.cat(self.u, dim=0)
        self.u = torch.reshape(tmp, (len(self.u), self.u[0].size()[0], self.u[0].size()[1]))
        print(f"size of u: {self.u.size()}")

        # Define attention vector selected randomly (256)
        attention_vector = torch.randn(self.hidden_dim * 2)  # u_w global context vector
        print(f"attention vector: {attention_vector.size()}")  # 256

        # Define attention weights (222, 32)
        tmp = torch.matmul(self.u, attention_vector)
        m = nn.Softmax(dim=0)
        a = m(tmp)
        print(f"attention and size: {a.size()}")

        # Se faccio prodott classico torch.matmul viene una matrice 3D che non credo sia quello che vogliamo
        # Allora ho provato seguendo il paper ha fare il prodotto tra ai e hi però boooooh

        # Compute s_i = a_i x self_hidden_states_i (222, 256)
        result = []
        for ai, hi in zip(a, self.hidden_states):
            result.append(torch.matmul(ai, hi))  # 32 x (32, 256)

        tmp = torch.cat(result, dim=0)
        s = torch.reshape(tmp, (len(result), result[0].size()[0]))
        print(f"s: {s.size()}")

        return s