import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange


def cumprod_exclusive(
  tensor: torch.Tensor
) -> torch.Tensor:
    r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod

def raw2outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor
    ):
    r"""
    Convert the raw NeRF output into RGB and other maps.
    """

    # δi = ti+1 − ti ---> (n_rays, n_samples)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Normalize encoded directions of each bin
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # αi = 1 − exp(−σiδi) ---> (n_rays, n_samples)
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3]) * dists)

    # Ti(1 − exp(−σiδi)) ---> (n_rays, n_samples)
    # cumprod_exclusive = a product of exponential values = exponent of sum of values
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    # Equation 3 in the paper
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    return rgb_map, weights
