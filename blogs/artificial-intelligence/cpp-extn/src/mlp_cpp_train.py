import math
from torch import nn
from torch.autograd import Function
import torch

import mlp_cpp

torch.manual_seed(42)


class MLPFunction(Function):
    @staticmethod
    def forward(ctx, input, hidden_weights, hidden_bias, output_weights, output_bias):
        output = mlp_cpp.forward(input, hidden_weights, hidden_bias, output_weights, output_bias)
        variables = [input, hidden_weights, hidden_bias, output_weights, output_bias]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_hidden_weights, grad_hidden_bias, grad_output_weights, grad_output_bias =  mlp_cpp.backward( *ctx.saved_variables, grad_output)
        return grad_input, grad_hidden_weights, grad_hidden_bias, grad_output_weights, grad_output_bias


class MLP(nn.Module):
    def __init__(self, input_features=5, hidden_features=15):
        super(MLP, self).__init__()
        self.input_features = input_features
        self.hidden_weights = nn.Parameter(torch.rand(hidden_features,input_features))
        self.hidden_bias = nn.Parameter(torch.rand(1, hidden_features))
        self.output_weights = nn.Parameter(torch.rand(1,hidden_features))
        self.output_bias = nn.Parameter(torch.rand(1, 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.001
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        return MLPFunction.apply(input, self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias)