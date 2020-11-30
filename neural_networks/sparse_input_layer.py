from os import path
from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.autograd.function as Function
import sys
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])


class SparseLinear(Function.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, sparse_input, weight, bias=None):
        sparse_input_np = csr_matrix(sparse_input.data.numpy())
        ctx.save_for_backward(sparse_input, weight, bias)
        weight_np = weight.data.numpy().T
        output = csr_matrix.dot(sparse_input_np, weight_np)
        output = torch.autograd.Variable(torch.from_numpy(output).float())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        sparse_input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        grad_output_np = grad_output.data.numpy()
        # print grad_output_np
        sparse_input_np = csr_matrix(sparse_input.data.numpy().T)
        # print sparse_input_np
        if ctx.needs_input_grad[1]:
            grad_weight = torch.autograd.Variable(torch.from_numpy((csr_matrix.dot(sparse_input_np, grad_output_np)).T).float())
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
