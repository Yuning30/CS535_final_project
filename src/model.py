import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, feature_size, hidden_size=16):
        super(encoder, self).__init__()

        self.encoder = nn.LSTM(input_size=feature_size, hidden_size=hidden_size)

    def forward(self, x):
        # return the hidden state
        lengths = [len(data) for data in x]
        padded_x = nn.utils.rnn.pad_sequence(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(padded_x, lengths)
        _, (h_n, c_n) = self.encoder(packed_x)
        return h_n


class decoder(nn.Module):
    def __init__(self, hidden_size=16):
        super(decoder, self).__init__()

        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.dense_layer = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.unrolled_steps = 64

    def forward(self, final_hidden_state):
        # return a sequence of probabilities
        replicates = [final_hidden_state.clone() for _ in range(0, self.unrolled_steps)]
        replicates = torch.stack(replicates)

        h_0 = final_hidden_state.clone()
        h_0 = torch.unsqueeze(h_0, 0)
        c_0 = torch.zeros_like(h_0)

        out, (_, _) = self.decoder(replicates, (h_0, c_0))
        out = self.dense_layer(out)
        probs = self.sigmoid(out)

        return probs
