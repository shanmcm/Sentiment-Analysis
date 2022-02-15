import torch
import torch.nn as nn
import torch.nn.functional as F

import params


class EasyNet(nn.ModuleList):

    def __init__(self, batch_size, hidden_dim, lstm_layers, embedding_dim):
        super(EasyNet, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        self.input_size = embedding_dim  # embedding dimention

        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(params.NUM_FEATURES, 128, bidirectional=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=257)
        self.fc2 = nn.Linear(257, 1)

    def forward(self, x):


        return out