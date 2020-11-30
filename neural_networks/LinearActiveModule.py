import sys
from os import path
import torch.nn as nn
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

class LinearActiveLayer(nn.Module):
    def __init__(self, input_size, out_size, linear_func, active_func):
        super(LinearActiveLayer, self).__init__()
        self.fc = linear_func(input_size, out_size, bias=True)
        self.act = active_func()

    def forward(self, x):
        return self.act(self.fc(x))

