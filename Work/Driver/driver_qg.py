import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import datetime
import numpy as np
import IPython.display as display
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, List
from tqdm.notebook import tqdm
import torch.optim as optim
import dataclasses
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os
import sys
import warnings
import importlib
from pathlib import Path
import importlib.util

warnings.filterwarnings("ignore", category=UserWarning)
    
# Create an argument parser
parser = argparse.ArgumentParser(description='Parse args')

# Add the run_num argument to specify the run number
parser.add_argument('--run_num', type=int, help='Run number for configuration file', required=True)

# Parse the command-line arguments
args = parser.parse_args()
run_number = args.run_num

sys_dir=f'/gdata/Results/mjmvega/QG/Run{run_number:05d}/'
sys.path.append(os.path.join(sys_dir, "Code"))

## Load config file
config_module = f'Config.Run{run_number:05d}'

from Utils.utils import print_config, git_info
print("Git hash info:", git_info())

try:
    config = importlib.import_module(config_module)
    print(f"Successfully loaded configuration from {config_module}")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print_config(config.params) 
except ModuleNotFoundError:
    print(f"Configuration file for run {run_num} not found.")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    raise

## Set-up grid
from Grid.grid import Grid
grid_DNS=Grid(config.params.grid.Lx,config.params.grid.Ly,config.params.grid.Nx,config.params.grid.Ny)

## Set-up spectral derivatives and operators
from Operators.operators import SpectralDerivatives, LinearOperator, NonlinearOperator
spec_deriv_DNS=SpectralDerivatives(grid_DNS)
linop_DNS = LinearOperator(spec_deriv_DNS,config.params.pde)
nonlinop_DNS = NonlinearOperator(spec_deriv_DNS,config.params.pde)

print(f"Successfully loaded derivatives and operators")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Set-up initial conditions and masks
config.params.ic.option = getattr(config.params.ic, 'option', 1) # Default case is random init with wavenumbers
if config.params.ic.option == 1:
    print(f"Using wavenumber ICs")
    from Initial_forcing.ics import init_randn
    init_conds_DNS =  init_randn(config.params.ic.energy, config.params.ic.wavenumbers, grid_DNS, spec_deriv_DNS, config.params.ic.seed)
elif config.params.ic.option == 2:
    from Initial_forcing.ics import init_fpc
    print(f"Using constant u-vel ICs")
    init_conds_DNS =  init_fpc(config.params.ic.init_vel, grid_DNS, spec_deriv_DNS, config.params.ic.seed)
elif config.params.ic.option == 3:
    from Initial_forcing.ics import init_closure
    print(f"Using closure ICs")
    init_conds_DNS =  init_closure(config.params.ic, grid_DNS, spec_deriv_DNS, config.params.ic.seed)
else:
    raise ValueError("Invalid IC option. Check config.")

# Smoothen sharp masks with a Gaussian filter (avoids Gibbs oscillations)
from Operators.spectral_conversion import to_physical, to_spectral
    
if config.params.mask.option == 1:
    print(f"Using circular mask")
    from Masks.masks import create_circular_mask_v2
    config.params.mask.x_center = getattr(config.params.mask, 'x_center', None)
    config.params.mask.y_center = getattr(config.params.mask, 'y_center', None)
    obstacle_mask_DNS =  create_circular_mask_v2(grid_DNS,config.params.mask.r,config.params.mask.x_center,
                                              config.params.mask.y_center,config.params.mask.tol)

elif config.params.mask.option == 2:
    print(f"Using cape v2")
    config.params.mask.x_scale = getattr(config.params.mask, 'x_scale', None)
    config.params.mask.y_scale = getattr(config.params.mask, 'y_scale', None)
    from Masks.masks import  create_cape_mask_v2
    obstacle_mask_DNS =   create_cape_mask_v2(grid_DNS,config.params.mask.r,config.params.mask.y_top, config.params.mask.x_center,
                                              config.params.mask.y_center, config.params.mask.y_width, 
                                              config.params.mask.x_left, config.params.mask.x_right, 
                                              config.params.mask.x_scale, config.params.mask.y_scale, config.params.mask.tol)
elif config.params.mask.option == 3:
    print(f"Using boundary mask")
    from Masks.masks import  create_boundary_mask
    obstacle_mask_DNS =   create_boundary_mask(grid_DNS,config.params.mask.width, config.params.mask.tol)
else:
    raise ValueError("Invalid mask option. Check config.")

print(f"Successfully created initial conditions and obstacles")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Set-up forcing
config.params.forcing.dynamic = getattr(config.params.forcing, 'dynamic', False) # Default case is not dynamic
if config.params.forcing.dynamic:
    print(f"Using dynamic forcing")
else:
    print(f"Not using dynamic forcing")

if config.params.forcing.option ==1:
    print(f"Using cos forcing")
    from Initial_forcing.forcing import cos_forcing
    forcing_DNS = cos_forcing
elif config.params.forcing.option ==2:
    print(f"Using sin forcing")
    from Initial_forcing.forcing import sin_forcing
    forcing_DNS = sin_forcing
elif config.params.forcing.option ==0:
    forcing_DNS = None
    config.params.forcing=None
else:
    raise ValueError("Invalid forcing option. Check config.")
    
## Set-up sponging
if config.params.sponge.option ==1:
    print(f"Using boundary sponging")
    from Masks.masks import create_sponge
    sponge_DNS = create_sponge(grid_DNS,config.params.sponge.x_right,config.params.sponge.x_left,config.params.sponge.y_top,
                               config.params.sponge.y_bottom,config.params.sponge.tol)
    # Smoothen sharp sponges with a Gaussian filter (avoids Gibbs oscillations)
    sponge_DNS_h=to_spectral(sponge_DNS)
    G_gaussian = torch.exp(-4*spec_deriv_DNS.krsq*((2*grid_DNS.dx)**2)/24)
    sponge_DNS_h = G_gaussian*sponge_DNS_h
    sponge_DNS=to_physical(sponge_DNS_h)
else:
    sponge_DNS = None
    
## Set-up closure
if config.params.pde.closure_option == 1:
    print(f"Using Leith closure")
elif config.params.pde.closure_option == 2:
    print(f"Using Smag closure")
elif config.params.pde.closure_option == 3:
    print(f"Using Clark closure")
elif config.params.pde.closure_option == 4:
    if config.params.pde.func_option == 1:
        print(f"Using Leith closure")
    else:
        print(f"Using Smag closure")     
    if config.params.pde.A[0] == 0 and config.params.pde.A[1] == 0:
        if config.params.pde.alpha is not None:
            print(f"Using Dynamic alpha model")
        else:
            raise ValueError("Invalid Dynamic model option. Check A or alpha.")
    elif config.params.pde.A[0] == 0:
        print(f"Using SDMM-1: Sequential Structural + Functional")      
    elif config.params.pde.A[1] == 0:
        print(f"Using SDMM-2: Sequential Functional + Structural")    
    else:
        print(f"Using 2-parameter DMM") 
elif config.params.pde.closure_option == 5:
    print(f"Using ML closure")
elif config.params.pde.closure_option == 6:
    print(f"Using hybrid ML  + Smag closure")
elif config.params.pde.closure_option == 7:
    print(f"Using  hybrid ML  + Leith closure")  
    
# Use splitting scheme
if config.params.pde.closure_option in (5,6,7):
    # Helper true DNS operators
    grid_DNS_512=Grid(config.params.grid.Lx,config.params.grid.Ly,512,512)
    spec_deriv_DNS_512=SpectralDerivatives(grid_DNS_512)
    nonlinop_DNS_512 = NonlinearOperator(spec_deriv_DNS_512,config.params.pde)
    
    # Load the model weights
    diff_dir = "/gdata/projects/ml_scope/Turbulence/Diffusion_V0002/Results"
    model_dir  = os.path.join(diff_dir, f"Run{config.params.sde.run_cond:05d}")  
    sys.path.append(os.path.join(model_dir, "Code"))
    from Models.networks import UNet_large
    model_cond = UNet_large(channels=config.params.diff.channels,in_channels=2,
                       out_channels=config.params.diff.out_channels,dropout_rate=config.params.diff.dropout_rate,
                      attention=config.params.diff.attention,condition=True).cuda()
    cond_model_load_directory = os.path.abspath(os.path.join(model_dir,f"Checkpoints/model_{config.params.sde.checkpoint:05d}.pt"))
    

    from Utils.utils import load_ddp_weights
    load_ddp_weights(model_cond, cond_model_load_directory)
    
    print(f"Cond model loaded successfully")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    H, W = grid_DNS_512.Nx, grid_DNS_512.Ny 
    template_x_init = torch.zeros(H, W, device="cuda")
    
    # Load model as a partial function of model_cond_data_t
    from functools import partial
    from Models.samplers import get_samples_multidiff
    model = partial(get_samples_multidiff,x_init=template_x_init, option=config.params.sde.option,
                    seed=config.params.sde.seed, diffusion_time_steps=config.params.sde.time_steps,
                    num_corrections=config.params.sde.num_corrections,model_uncond=None,
                    model_cond=model_cond, corr_scale=config.params.sde.corr_scale,
                    multidiff=config.params.sde.multidiff,sdedit=config.params.sde.sdedit,
                    uncond_flag=config.params.sde.model_uncond_flag, 
                    CFG=config.params.sde.CFG,save_traj=config.params.sde.save_traj)

    ## Run simulation
    print(f"Simulation started")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    from Simulation.simulation_split_closure_ML import Simulation
    sim_DNS = Simulation(grid_DNS,config.params.pde,spec_deriv_DNS,linop_DNS,nonlinop_DNS,init_conds_DNS,config.params.time,
                         config.params.forcing, model, nonlinop_DNS_512, config.params.diff.field_std,
                         obstacle_mask_DNS,forcing_DNS, sponge_DNS)
else:
    from Simulation.simulation_split_closure import Simulation
    sim_DNS = Simulation(grid_DNS,config.params.pde,spec_deriv_DNS,linop_DNS,nonlinop_DNS,init_conds_DNS,config.params.time,
                         config.params.forcing,obstacle_mask_DNS,forcing_DNS, sponge_DNS)

with torch.no_grad(): 
    solution_field = sim_DNS.run()

if config.params.pde.closure_option in (5,6,7):
    q_DNS = sim_DNS.get_upsampled_vorticity()
else:    
    dyn_closure_vals = sim_DNS.get_dynamic_coeffs()

print(f"Simulation completed successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Save numpy files
from Utils.utils import save_file
save_file(grid_DNS,spec_deriv_DNS,solution_field,run_number,config.params.time, sys_dir)

if config.params.pde.closure_option in (5,6,7):
    from Utils.utils import save_q_DNS
    save_q_DNS(q_DNS,run_number,sys_dir)
else:  
    from Utils.utils import save_dynamic_coeffs
    save_dynamic_coeffs(dyn_closure_vals,run_number,sys_dir)

print(f"Simulation np & pt files and plots saved successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Save spectrum
from Utils.utils import save_spectrum_plots

save_spectrum_plots(solution_field,spec_deriv_DNS,run_number,config.params.time, sys_dir)

print(f"Simulation spectrum plots saved successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
