'''
To do:
- correct forget gate function
- update embedding with ELMO
'''

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import params  # aggiunto params file in cui salviamo tutti gli hyperparameters
from lstm_cell import LSTMCell  # importo la notra LSTMCell
from attention_layer import AttentionLayer


class SentimentAnalysis(nn.ModuleList):
    def __init__(self, batch_size, hidden_dim, embedding_size,
                 dropout_rate):  # aggiunto parametri che prima erano in args
        super(SentimentAnalysis, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.input_size = embedding_size
        self.num_classes = 5
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Bi-LSTM
        # Forward and backward
        self.lstm_cell_forward = LSTMCell(self.input_size, self.hidden_dim, self.hidden_dim)  # x,h,c
        self.lstm_cell_backward = LSTMCell(self.input_size, self.hidden_dim, self.hidden_dim)  # x,h,c
        # LSTM layer
        self.lstm_cell = LSTMCell(self.hidden_dim * 2, self.hidden_dim * 2,
                                  self.hidden_dim * 2)

        # Linear layer
        self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)

        # Attention parameters
        self.hidden_states_lstm = []
        self.attention = AttentionLayer(self.hidden_dim)

    def forward(self, x):
        hs_forward = torch.zeros(x.size(0), self.hidden_dim)
        cs_forward = torch.zeros(x.size(0), self.hidden_dim)
        hs_backward = torch.zeros(x.size(0), self.hidden_dim)
        cs_backward = torch.zeros(x.size(0), self.hidden_dim)

        hs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)
        cs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)

        # Weights initialization
        torch.nn.init.kaiming_normal_(hs_forward)
        torch.nn.init.kaiming_normal_(cs_forward)
        torch.nn.init.kaiming_normal_(hs_backward)
        torch.nn.init.kaiming_normal_(cs_backward)
        torch.nn.init.kaiming_normal_(hs_lstm)
        torch.nn.init.kaiming_normal_(cs_lstm)

        forward = []
        backward = []

        # Unfolding Bi-LSTM
        # Forward
        for i in range(x.size(1)):
            inp = x[:, i, :].clone()
            hs_forward, cs_forward = self.lstm_cell_forward(inp, (hs_forward, cs_forward))
            forward = forward + [hs_forward]
        # Backward
        for i in reversed(range(x.size(1))):
            inp = x[:, i, :].clone()
            hs_backward, cs_backward = self.lstm_cell_backward(inp, (hs_backward, cs_backward))
            backward = backward + [hs_backward]
        # LSTM
        for fwd, bwd in zip(forward, backward):
            input_tensor = torch.cat((fwd, bwd), 1)
            hs_lstm, cs_lstm = self.lstm_cell(input_tensor, (hs_lstm, cs_lstm))
            self.hidden_states_lstm.append(hs_lstm)

        hs_with_attention = self.attention(self.hidden_states_lstm)
        # Last hidden state is passed through a linear layer
        out = self.linear(hs_with_attention)

        return out
