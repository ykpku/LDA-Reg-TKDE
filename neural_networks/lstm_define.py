from os import path
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layer=1, drop_out=0, use_gpu=False):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, dropout=drop_out)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.use_gpu = use_gpu

    def forward(self, x):
        # Set initial states
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = F.sigmoid(self.fc(out[:, -1, :]))
        return out
