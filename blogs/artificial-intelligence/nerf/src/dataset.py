import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange


class NerfDataset(torch.utils.data.Dataset):
    def __init__(self, s=0, e=100):
        data = np.load('tiny_nerf_data.npz')
        self.images = torch.from_numpy(data['images'])[s:e]
        self.poses = torch.from_numpy(data['poses'])[s:e] # 4*4 Rotational matrix to change camera coordinates to NDC(Normalized Device Coordinates)
        self.focal = torch.from_numpy(data['focal']).item()
        
    def get_rays(self, pose_c2w, height=100.0, width=100.0, focal_length = 138):
        # Apply pinhole camera model to gather directions at each pixel
        i, j = torch.meshgrid(torch.arange(width),torch.arange(height),indexing='xy')
        directions = torch.stack([(i - width * .5) / focal_length,
                                -(j - height * .5) / focal_length,
                                -torch.ones_like(i) #-ve is not necessary
                               ], dim=-1)

        # Apply camera pose to directions
        product = directions[..., None, :] * pose_c2w[:3, :3] #(W, H, 3, 3)
        rays_d = torch.sum(product, dim=-1) #(W, H, 3)

        # Origin is same for all directions (the optical center)
        rays_o = pose_c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d
        
    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        rays_o, rays_d = self.get_rays(pose, focal_length=self.focal)
        return rays_o, rays_d, image
        

def encode(pts, num_freqs):
    freq = 2.**torch.linspace(0, num_freqs - 1, num_freqs)
    encoded_pts = []
    for i in freq:
        encoded_pts.append(torch.sin(pts*i))
        encoded_pts.append(torch.cos(pts*i))
    return torch.concat(encoded_pts,dim=-1)

def encode_pts(pts, L=10):
    flattened_pts = pts.reshape(-1,3)
    return encode(flattened_pts, L)

def encode_dirs(dirs, n_samples=64, L=4):
    #normalize before encode
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True) #(W,H,3)
    dirs = dirs[..., None, :].expand(dirs.shape[:-1]+(n_samples,dirs.shape[-1])) #(W,H,num_samples,3)
    #print(dirs.shape)
    flattened_dirs = dirs.reshape((-1, 3))
    return encode(flattened_dirs,L)
   
def stratified_sampling(rays_o, rays_d, n_samples=64, perturb=0.2):
    # rays_o, rays_d = self.get_rays(pose)
    z_vals = torch.linspace(2,6-perturb, n_samples) + torch.rand(n_samples)*perturb
    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples]).to('cuda') #(W,H,n_samples)

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] #(W,H,n_samples,3)
    return pts.to('cuda'), z_vals.to('cuda')
    
def sample_pdf(
  bins: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
) -> torch.Tensor:
    r"""
    Apply inverse transform sampling to a weighted set of points.
    """

    # Normalize weights to get PDF.
    # weights ---> [n_rays, n_samples]
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, n_samples]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1) # [n_rays, n_samples]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, n_samples + 1]

    # Sample random weights uniformly and find their indices in the CDF distribution
    u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]
    inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

    # Stack consecutive indices as pairs
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

    # Collect new weights from cdf and new bins from existing bins.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]] # [n_rays, n_samples, n_samples + 1]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                       index=inds_g) # [n_rays, n_samples, 2]
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g) # [n_rays, n_samples, 2]

    # Normalize new weights and generate heirarchical samples from new bins.
    denom = (cdf_g[..., 1] - cdf_g[..., 0]) # [n_rays, n_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]


def hierarchical_sampling(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  z_vals: torch.Tensor,
  weights: torch.Tensor,
  n_samples_hierarchical: int,
  perturb: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Apply hierarchical sampling to the rays.
    """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples_hierarchical,
                          perturb=perturb)

    # Rescale the points using rays_o and rays_d
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples_hierarchical, 3]
    return pts.to('cuda'), z_vals_combined.to('cuda'), new_z_samples.to('cuda')
