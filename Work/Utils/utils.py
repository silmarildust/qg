import importlib
import numpy as np
import math
import matplotlib.pyplot as plt
from Plotting.plots import vorticity_plots, spectrum_plot, spectrum
from Operators.spectral_conversion import to_physical, to_spectral, dealias, get_puv
import os
import torch
import pickle
import shutil
import subprocess
from collections import OrderedDict

def print_config(obj, indent=0):
    """Recursively prints all attributes of a class or object."""
    if not hasattr(obj, "__dict__") and not isinstance(obj, type):  # If it's a simple value, print it
        print(" " * indent + str(obj))
        return

    for attr_name in dir(obj):
        if attr_name.startswith("__"):  # Skip special attributes
            continue

        attr_value = getattr(obj, attr_name)

        if isinstance(attr_value, type):  # If it's a class, recurse into it
            print(" " * indent + f"{attr_name}:")
            print_config(attr_value, indent + 4)
        elif not callable(attr_value):  # Print regular attributes
            print(" " * indent + f"{attr_name} = {attr_value}")
            
def git_info():
    def run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""
    return {
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "no-branch",
        "commit": run(["git", "rev-parse", "HEAD"]) or "no-git",
        "describe": run(["git", "describe", "--always", "--dirty", "--tags"]) or "",
    }

def load_ddp_weights(model, ckpt_path, device="cpu", strict=True):
    state_dict = torch.load(ckpt_path, map_location=device)
    # remove "module." prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state_dict[name] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    print("Loaded checkpoint:", ckpt_path)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

def save_file(grid,spectral_derivative,solution_field, run_number, time_params, save_dir):
    file_name = f'fields_Run{run_number:05d}.npy'
    file_path = os.path.join(save_dir, file_name)
    np.save(file_path, solution_field.cpu().numpy())   
          
    save_path = os.path.join(save_dir, 'Plots')
    os.makedirs(save_path, exist_ok=True)
    
    for timestep in range(solution_field.shape[2]):
        fig, ax = vorticity_plots(grid,solution_field, timestep, time_params)

        # Save the plot as an image 
        time = timestep * time_params.save_int * time_params.dt
        plot_file_name = f'vorticity_Run{run_number:05d}_t_{time:06.3f}.png'
        plot_file_path = os.path.join(save_path, plot_file_name)
        fig.savefig(plot_file_path,bbox_inches='tight',dpi=300)  # Save the plot as an image
        plt.close(fig)  # Close the figure to free memory
    
def save_spectrum_plots(solution_field, spectral_derivative, run_number, time_params, save_dir):
    save_path = os.path.join(save_dir, 'Spectrum')
    os.makedirs(save_path, exist_ok=True)
    
    for timestep in range(solution_field.shape[2]):
        
        qh_sol=to_spectral(solution_field[:,:,timestep,0].squeeze()).cuda() # Extract just vorticity field
        ph_sol,uh_sol,vh_sol= get_puv(qh_sol, spectral_derivative)
        
        z = torch.abs(qh_sol)**2 # Get enstrophy
        e = torch.abs(uh_sol)**2 + torch.abs(vh_sol)**2 # Get kinetic energy
        
        k,[ek,zk]=spectrum([e,z],spectral_derivative)

        fig, ax =  spectrum_plot(k, ek, zk, timestep, time_params)

        # Save the plot as an image
        time = timestep * time_params.save_int * time_params.dt
        plot_file_name = f'spectrum_Run{run_number:05d}_t_{time:06.3f}.png'
        plot_file_path = os.path.join(save_path, plot_file_name)
        fig.savefig(plot_file_path,bbox_inches='tight',dpi=300)  # Save the plot as an image
        plt.close(fig)  # Close the figure to free memory
        
def save_dynamic_coeffs(dynamic_coeffs, run_number, save_dir):
    if dynamic_coeffs is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    file_name = f'dynamic_coeffs_Run{run_number:05d}.npy'
    file_path = os.path.join(save_dir, file_name)

    np.save(file_path, np.array(dynamic_coeffs))
    
def save_q_DNS(q_DNS, run_number, save_dir):
    if q_DNS is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    file_name = f'superresolved_q_Run{run_number:05d}.npy'
    file_path = os.path.join(save_dir, file_name)

    np.save(file_path, np.array(q_DNS))