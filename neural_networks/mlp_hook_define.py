from os import path
import torch.nn as nn
import sys
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
from neural_networks.SparseLinearModule import SparseLinearModule
from neural_networks.LinearActiveModule import LinearActiveLayer


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layer, sparse=False):
        super(Net, self).__init__()
        if sparse:
            self.layer1 = LinearActiveLayer(input_size, hidden_size, SparseLinearModule, nn.Sigmoid)
        else:
            self.layer1 = LinearActiveLayer(input_size, hidden_size, nn.Linear, nn.Sigmoid)
        self.num_layer = num_layer

        if self.num_layer == 2:
            if sparse:
                self.layer2 = LinearActiveLayer(hidden_size, hidden_size, SparseLinearModule, nn.Sigmoid)
            else:
                self.layer2 = LinearActiveLayer(hidden_size, hidden_size, nn.Linear, nn.Sigmoid)

        self.layer_o = LinearActiveLayer(hidden_size, num_classes, nn.Linear, nn.Sigmoid)

    def forward(self, x):
        out = self.layer1(x)
        if self.num_layer == 2:
            out = self.layer2(out)
        out = self.layer_o(out)
        return out