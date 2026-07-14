import torch
import torch.nn.functional as F
import numpy as np
from Operators.spectral_conversion import to_physical, to_spectral, dealias, get_puv
from Grid.grid import Grid
from Operators.operators import SpectralDerivatives

def int_sq(y, grid):
    """
    Uses Parseval's theorem to find \int_y^2 dA
    """
    Y = torch.sum(torch.abs(y[:, 0])**2) + 2*torch.sum(torch.abs(y[:, 1:])**2)
    n = grid.Lx * grid.Ly  # Use grid object for Lx and Ly
    return Y * n

def int_cross_sq(x,y, grid):
    """
    Uses Parseval's theorem to find \int_x*y dA
    """
    XY_0 = torch.sum(x[:, 0]*torch.conj(y[:, 0])) 
    XY_1 = torch.sum(x[:, 1:]*torch.conj(y[:, 1:])) 
    
    XY = torch.real(XY_0 + 2* XY_1)
    
    n = grid.Lx * grid.Ly  # Use grid object for Lx and Ly
    return XY * n

def init_randn(energy, wavenumbers, grid, spectral_derivative, seed=86):
    """
    Generates initial conditions based on specified energy and wavenumber limits
    """
    torch.manual_seed(seed)
    
    # Use spectral_derivative for kr, ky, and krsq
    K = torch.sqrt(spectral_derivative.krsq)  # Wavenumber of each point in frequency space
    k = spectral_derivative.kr.repeat(grid.Ny, 1)  # Ensure proper shape for k

    # Generate random complex field in spectral space
    qih = torch.randn(spectral_derivative.krsq.size(), dtype=torch.complex128).to(grid.device)
    
    # Apply wavenumber filters
    qih[K < wavenumbers[0]] = 0.0
    qih[K > wavenumbers[1]] = 0.0
    qih[k == 0.0] = 0.0  # Handle zero wavenumber 
    
    # Normalize initial condition energy
    E0 = energy
    Ei = 0.5 * (int_sq(spectral_derivative.kr * spectral_derivative.irsq * qih, grid) +
                int_sq(spectral_derivative.ky * spectral_derivative.irsq * qih, grid)) / (grid.Lx * grid.Ly)
    
    # Scale to the desired energy
    qih *= torch.sqrt(E0 / Ei)
    
    pih,uih,vih = get_puv(qih,spectral_derivative)
    
    return qih,pih,uih,vih

def init_fpc(init_vel, grid, spectral_derivative, seed=86):
    """
    Generates inlet initial conditions (constant background velocities)
    """
    torch.manual_seed(seed)
    
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, steps=Nx, device= grid.device)
    y = torch.linspace(0, Ly, steps=Ny, device= grid.device)

    # Create meshgrid for x, y
    X = x.unsqueeze(0).expand(Ny, Nx)
    Y = y.unsqueeze(1).expand(Ny, Nx)
    
    # Initial vorticity, streamfunction and v-velocity
    qih = torch.zeros_like(spectral_derivative.krsq, dtype=torch.complex128).to(grid.device)
    
    pih,uih,vih = get_puv(qih,spectral_derivative)

    # In spectral space
    uih[0, 0] = init_vel 
    
    return qih,pih,uih,vih

def init_closure(ic_params, grid, spectral_derivative, seed=86):
    """
    Generates downsampled initial conditions from DNS initial conditions
    """
    DNS_data = torch.from_numpy(np.load(ic_params.file_path)).float().to(grid.device)
    DNS_grid=Grid(Nx=ic_params.Nx,Ny=ic_params.Ny)
    spec_deriv_DNS=SpectralDerivatives(DNS_grid)

    qh=to_spectral(DNS_data[...,ic_params.start_time,0].squeeze())
    
    del DNS_data

    G = torch.exp(-4*spec_deriv_DNS.krsq*((ic_params.scale*DNS_grid.dx)**2)/24)
    qh_dr = G*qh
    G = spec_deriv_DNS.ky.abs()<(torch.pi/(ic_params.scale*DNS_grid.dx))
    qh_dr = G*qh_dr
    G =  spec_deriv_DNS.kr.abs()<(torch.pi/(ic_params.scale*DNS_grid.dx))
    q_dr = to_physical(G*qh_dr)
    
    qi = F.avg_pool2d(q_dr.unsqueeze(0).unsqueeze(0), kernel_size=ic_params.scale, stride=ic_params.scale).squeeze(0).squeeze(0)
    qih = to_spectral(qi)
    
    pih,uih,vih = get_puv(qih,spectral_derivative)
    
    return qih,pih,uih,vih