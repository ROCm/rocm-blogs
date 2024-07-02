import torch
import sys
from ..rocm import bmm_abe_i8, bmm_ab_i8_e_f32

class BMM_ABE_I8(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer('a', torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int8

        return bmm_abe_i8(a, b, self.a.item())

    @staticmethod
    def from_scale(a_scale, b_scale, output_scale):
        bmm_module = BMM_ABE_I8(1.0)
        alpha = a_scale * b_scale / output_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module

class BMM_AB_I8_E_F32(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer('a', torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] Float32
        y = bmm_ab_i8_e_f32(a, b, self.a.item())
        return y

    @staticmethod
    def from_scale(a_scale, b_scale):
        bmm_module = BMM_AB_I8_E_F32(1.0)
        alpha = a_scale * b_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module
