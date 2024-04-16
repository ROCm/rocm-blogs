import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)


class MLP(torch.nn.Module):
    def __init__(self, input_features=5, hidden_features=15):
        super(MLP, self).__init__()
        self.input_features = input_features
        self.hidden_layer = torch.nn.Linear(input_features,hidden_features)
        self.output_layer = torch.nn.Linear(hidden_features,1)
        self.relu = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.001
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        out = self.relu(self.hidden_layer(input))
        out = self.output_layer(out)
        return out
        