import numpy as np
import matplotlib.pyplot as plt
import torch
import math


def vorticity_plots(grid,field,timestep, time_params):
    fig, ax = plt.subplots()  # Create a figure and axis

    cax = ax.imshow(field[:, :, timestep,0], cmap='seismic', origin='lower', vmax=10, vmin=-10, 
                    extent=[0, grid.Lx, 0, grid.Ly])  # 0 is vorticity

    plt.colorbar(cax, ax=ax)

    tick_positions = np.linspace(0, grid.Ly, 3)  # Creates ticks at 0, π, 2π
    tick_labels = [f"${pos:.2f}$" for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    # Set title
    ax.set_title(f"$\omega$ at T= {timestep} x{time_params.save_int} dt")

    return fig, ax  # Return the figure and axis objects

def spectrum_plot(k, ek, zk, timestep, time_params):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy spectrum (left plot)
    axes[0].loglog(k, ek)
    axes[0].set_xlim(1, 0.75*max(k))
    axes[0].set_ylim(10**(-12), 10**(-0.5))
    axes[0].set_aspect(0.20, adjustable='box')
    axes[0].set_title(f"Energy Spectrum at T= {timestep} x{time_params.save_int} dt")
    axes[0].set_xlabel("Wavenumber k")
    axes[0].set_ylabel("Energy Spectrum E(k)")
    
    # Enstrophy spectrum (right plot)
    axes[1].loglog(k, zk)
    axes[1].set_xlim(1, 0.75*max(k))
    axes[1].set_ylim(10**(-12), 10**(-0.5))
    axes[1].set_aspect(0.20, adjustable='box')
    axes[1].set_title(f"Enstrophy Spectrum at T= {timestep} x{time_params.save_int} dt")
    axes[1].set_xlabel("Wavenumber k")
    axes[1].set_ylabel("Enstrophy Spectrum Z(k)")
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return fig, axes  # Return the figure and axis objects


## Calculate the spectrum
def spectrum(y, spectral_derivative): 
    # y is in spectral space
    K = torch.sqrt(spectral_derivative.krsq)
    d = 0.5
    k = torch.arange(1, int(K[-1,-1])-1)
    m = torch.zeros(k.size())
    
    e = [torch.zeros(k.size()) for _ in range(len(y))]
    for ik in range(len(k)):
        n = k[ik]
        i = torch.nonzero((K < (n + d)) & (K > (n - d)), as_tuple=True)
        m[ik] = i[0].numel()
        for j, yj in enumerate(y):
            e[j][ik] = torch.sum(yj[i]) * k[ik] * math.pi / (m[ik] - d)
    return k, e