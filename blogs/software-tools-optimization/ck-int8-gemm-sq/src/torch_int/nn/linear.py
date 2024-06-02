import torch
import sys
from ..rocm import (linear_ab_i8_de_f32,
                    linear_abde_i8,
                    linear_relu_abde_i8
                    )
from ..functional.quantization import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
)

class Linear_ABDE_I8(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features, self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros((1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_abde_i8(x, self.weight, self.bias, self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = Linear_ABDE_I8(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module

class Linear_ReLU_ABDE_I8(torch.nn.Module):
    # For fc1
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features, self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros((1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_relu_abde_i8(x, self.weight, self.bias, self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        # TODO: add zero-point to prevent the bit waste
        int8_module = Linear_ReLU_ABDE_I8(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module
    
class Linear_AB_I8_DE_F32(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features, self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros((1, self.out_features), dtype=torch.float32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_ab_i8_de_f32(x, self.weight, self.bias, self.a.item(), 1.0)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = Linear_AB_I8_DE_F32(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        int8_module.bias = module.bias.to(torch.float32)
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale
        return int8_module


