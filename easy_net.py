import torch
import torch.nn as nn
import torch.nn.functional as F


class EasyNet(nn.ModuleList):

    def __init__(self, batch_size, hidden_dim, lstm_layers, embedding_dim):
        super(EasyNet, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        self.input_size = embedding_dim  # embedding dimention

        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=257)
        self.fc2 = nn.Linear(257, 1)

    def forward(self, x):
        h = torch.zeros((x.size()[1], x.size(0), self.hidden_dim))
        c = torch.zeros((x.size()[1], x.size(0), self.hidden_dim))

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out, (hidden, cell) = self.lstm(x, (h, c))
        out = self.dropout(out)
        out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))

        return out