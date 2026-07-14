import torch
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Parse args
parser = argparse.ArgumentParser(description="Generate fDNS data from DNS runs.")
parser.add_argument("--scale", type=int, required=True, help="Downscale factor (e.g., 2, 4, 8)")
parser.add_argument("--filtering", action="store_true", help="Enable spectral filtering")
parser.add_argument("--interleaving", action="store_true", help="Interleaves back to fine grid")
parser.add_argument("--num_start", type=int, required=True, help="Start run index (inclusive)")
parser.add_argument("--num_end", type=int, required=True, help="End run index (non-inclusive)")
args = parser.parse_args()
scale = args.scale
filtering = args.filtering
interleaving= args.interleaving
num_start = args.num_start
num_end = args.num_end

project_root = '/gdata/projects/ml_scope/Turbulence/QG_V0003/Src'
if project_root not in sys.path:
    sys.path.append(project_root)

from Operators.spectral_conversion import to_spectral, to_physical, get_puv
from Grid.grid import Grid
DNS_grid = Grid(Nx=1024, Ny=1024)
LES_grid = Grid(Nx=1024//scale, Ny=1024//scale)

from Operators.operators import SpectralDerivatives
DNS_op = SpectralDerivatives(DNS_grid)
LES_op = SpectralDerivatives(LES_grid)

def LES_downscale(x_in,grid,spec_deriv,scale,filtering):
    if filtering:
        qh=to_spectral(x_in.squeeze())
        G = torch.exp(-4*spec_deriv.krsq*((scale*grid.dx)**2)/24)
        qh_dr = G*qh
        G = spec_deriv.ky.abs()<(torch.pi/(scale*grid.dx))
        qh_dr = G*qh_dr
        G =  spec_deriv.kr.abs()<(torch.pi/(scale*grid.dx))
        q_dr = to_physical(G*qh_dr)
    else:
        qh=to_spectral(x_in.squeeze())      
        G = spec_deriv.ky.abs()<(torch.pi/(scale*grid.dx))
        qh_dr = G*qh
        G =  spec_deriv.kr.abs()<(torch.pi/(scale*grid.dx))
        q_dr = to_physical(G*qh_dr)
    data_les = F.avg_pool2d(q_dr.unsqueeze(0).unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0).squeeze(0) 
#     data_les_interleaved = data_les.repeat_interleave(scale,dim=0).repeat_interleave(scale,dim=1)
    data_les_interleaved = q_dr
    return data_les,data_les_interleaved

with torch.no_grad():
    for i in range(num_start,num_end):
        file_name = f'/gdata/projects/ml_scope/Turbulence/QG_V0003/Results/Run{i:05d}/fields_Run{i:05d}.npy'
        DNS_data = torch.from_numpy(np.load(file_name)).float().cuda()
        if interleaving:
            LES_data = torch.zeros_like(DNS_data).float().cuda()
        else:
            LES_data = torch.zeros([DNS_data.shape[0]//scale,DNS_data.shape[1]//scale,DNS_data.shape[2],DNS_data.shape[3]]).float().cuda()

        for t in range(DNS_data.shape[2]):
            if interleaving:
                _,LES_data_tmp = LES_downscale(DNS_data[...,t,0],DNS_grid,DNS_op,scale,filtering)
                p_temp,u_temp,v_temp = get_puv(to_spectral(LES_data_tmp),DNS_op)
            else:
                LES_data_tmp,_ = LES_downscale(DNS_data[...,t,0],DNS_grid,DNS_op,scale,filtering)
                p_temp,u_temp,v_temp = get_puv(to_spectral(LES_data_tmp),LES_op)
            LES_data[:,:,t,0] = LES_data_tmp
            LES_data[:,:,t,1] = to_physical(p_temp)
            LES_data[:,:,t,2] = to_physical(u_temp)
            LES_data[:,:,t,3] = to_physical(v_temp)

        LES_np = LES_data.cpu().numpy()
#         save_dir  = os.path.dirname(file_name)
        save_dir = os.path.join(os.path.dirname(file_name), "PP")
        ds_size = 1024 // scale   
        pooling_name = "fields_pooled1024" if interleaving else "fields_1024"
        save_name = f'{pooling_name}_ds{ds_size}_Run{i:05d}.npy'
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, LES_np)