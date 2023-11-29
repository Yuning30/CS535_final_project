import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, feature_size, hidden_size=16):
        super(encoder, self).__init__()

        self.encoder = nn.LSTM(input_size=feature_size, hidden_size=hidden_size)

    def forward(self, x):
        # return the hidden state
        pass


class decoder(nn.Module):
    def __init__(self, hidden_size=16):
        super(decoder, self).__init__()

        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.dense_layer = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # return a sequence of probabilities
        pass
