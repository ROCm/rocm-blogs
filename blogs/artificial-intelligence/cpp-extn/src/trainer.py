from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch
torch.manual_seed(42)

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}
TIME_PRINT = {'s': 'seconds (s)', 'ms': 'milliseconds (ms)', 'us': 'microseconds (us)'}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp'])
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-f', '--features', type=int, default=32)
parser.add_argument('-hf', '--hidden_features', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=100000)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='ms')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from mlp_train import MLP
elif options.example == 'cpp':
    from mlp_cpp_train import MLP

device = torch.device("cuda")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
X = torch.randn(options.batch_size, options.features, **kwargs)
rnn = MLP(options.features, options.hidden_features).to(device, dtype)

# Force CUDA initialization
out = rnn(X)
out.sum().backward()

forward_time = 0
backward_time = 0
for _ in range(options.runs):
    rnn.zero_grad()

    start = time.time()
    out = rnn(X)
    elapsed = time.time() - start
    forward_time += elapsed

    start = time.time()
    out.sum().backward()
    elapsed = time.time() - start
    backward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_average = forward_time / options.runs * scale
backward_average = backward_time / options.runs * scale

print('Forward: {0:.3f} {2} | Backward {1:.3f} {2}'.format(
    forward_average, backward_average,
    TIME_PRINT[options.scale]))
