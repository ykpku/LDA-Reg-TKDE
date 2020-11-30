from os import path
import torch.nn as nn
import sys
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
from neural_networks.SparseLinearModule import SparseLinearModule


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layer, sparse=False):
        super(Net, self).__init__()
        if sparse:
            self.fc1 = SparseLinearModule(input_size, hidden_size, bias=True)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.sig = nn.Sigmoid()
        self.num_layer = num_layer

        if self.num_layer == 2:
            if sparse:
                self.fc3 = SparseLinearModule(hidden_size, hidden_size, bias=True)
            else:
                self.fc3 = nn.Linear(hidden_size, hidden_size, bias=True)

        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)
        self.sig_o = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sig(out)
        # for i in range(self.num_layer-1):
        #     out = self.hidden_list[i](out)
        #     out = self.sig(out)
        if self.num_layer == 2:
            out = self.fc3(out)
            out = self.sig(out)
        out = self.fc2(out)
        out = self.sig_o(out)
        return out