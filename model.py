import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 1000, num_layers = 6, dropout = 0.1, batch_first = True)
        self.linear = nn.Linear(1000, 1)

    def forward(self, x):
        h_0 = torch.zeros([6, x.shape[0], 1000], device = x.device) # initial hidden state: (layers number, batch size, hidden size)
        c_0 = torch.zeros([6, x.shape[0], 1000], device = x.device) # cell state: 

        out, _ = self.lstm(x, (h_0.detach(), c_0.detach()))
        out = self.linear(out[:, -1, :])

        return out
        