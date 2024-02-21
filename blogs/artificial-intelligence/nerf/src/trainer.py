import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange

from dataset import stratified_sampling, encode_pts, encode_dirs, hierarchical_sampling, NerfDataset
from render import raw2outputs
from model import NeRF

from random import shuffle


def nerf_forward(rays_o,rays_d,coarse_model,fine_model = None,n_samples=64):
    """
    Compute forward pass through model(s).
    """
    ################################################################################
    # Coarse model pass
    ################################################################################
    # Sample query points along each ray.
    query_points, z_vals = stratified_sampling(rays_o, rays_d, n_samples=n_samples)
    encoded_points = encode_pts(query_points) # (W*H*n_samples, 60)
    encoded_dirs = encode_dirs(rays_d) # (W*H*n_samples, 24)
    raw = coarse_model(encoded_points, viewdirs=encoded_dirs)
    raw = raw.reshape(-1,n_samples,raw.shape[-1])
    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, weights = raw2outputs(raw, z_vals, rays_d)
    outputs = {
      'z_vals_stratified': z_vals,
      'rgb_map_0': rgb_map
    }
    ################################################################################
    # Fine model pass
    ################################################################################
    # Apply hierarchical sampling for fine query points.
    query_points, z_vals_combined, z_hierarch = hierarchical_sampling(
      rays_o, rays_d, z_vals, weights, n_samples_hierarchical=n_samples)
    # Forward pass new samples through fine model.
    fine_model = fine_model if fine_model is not None else coarse_model
    encoded_points = encode_pts(query_points)
    encoded_dirs = encode_dirs(rays_d,n_samples = n_samples*2)
    raw = fine_model(encoded_points, viewdirs=encoded_dirs)
    raw = raw.reshape(-1,n_samples*2,raw.shape[-1]) #(W*H, n_samples*2, 3+1)
    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, weights = raw2outputs(raw, z_vals_combined, rays_d.reshape(-1, 3))

    # Store outputs.
    outputs['z_vals_hierarchical'] = z_hierarch
    outputs['rgb_map'] = rgb_map
    outputs['weights'] = weights
    return outputs

def main(iters=361, lr=5e-4):
    dataset = NerfDataset()
    test_dataset = NerfDataset(100,-1)
    coarse_model, fine_model = NeRF(), NeRF()
    coarse_model, fine_model = coarse_model.to('cuda'), fine_model.to('cuda')
    optimizer = torch.optim.Adam(list(coarse_model.parameters())+list(fine_model.parameters()), lr=lr)
    
    train_psnrs=[]
    valid_psnrs=[]
    
    lis = list(range(100))
    print('Starting training...')
    for iter in range(iters):
        coarse_model.train()
        for idx in lis:
            rays_o, rays_d, target_img = dataset[idx] # [100, 100, 3], [100, 100, 3], [100, 100, 3]
            rays_o, rays_d, target_img = rays_o.to('cuda'), rays_d.to('cuda'), target_img.to('cuda')
            outputs = nerf_forward(rays_o.reshape(-1,3), rays_d.reshape(-1,3), coarse_model, fine_model)
            rgb_predicted = outputs['rgb_map'].reshape(rays_o.shape)
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            psnr = -10. * torch.log10(loss)
            train_psnrs.append(psnr.item())
            print(f'Iter: {iter}, loss: {loss.item()}, psnr: {psnr}, idx:{idx}')
        torch.save({'coarse':coarse_model, 'fine':fine_model}, f'./checkpoints/{iter}.pt')
        shuffle(lis)
        
        ##### validate #####
        if iter%100==0:
            coarse_model.eval()
            rays_o, rays_d, target_img = test_dataset[0] # [100, 100, 3], [100, 100, 3], [100, 100, 3]
            rays_o, rays_d, target_img = rays_o.to('cuda'), rays_d.to('cuda'), target_img.to('cuda')
            outputs = nerf_forward(rays_o.reshape(-1,3), rays_d.reshape(-1,3), coarse_model, fine_model)
            rgb_predicted = outputs['rgb_map'].reshape(rays_o.shape)
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            psnr = -10. * torch.log10(loss)
            valid_psnrs.append(psnr.item())
            print(f'Testing, val_loss: {loss.item()}, psnr: {psnr}')

if __name__=="__main__": 
    main()            
            
        
