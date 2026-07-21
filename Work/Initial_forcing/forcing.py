import torch
import numpy as np
from Operators.spectral_conversion import to_physical, to_spectral, dealias
from Initial_forcing.ics import int_cross_sq

def cos_forcing(grid,spectral_derivative,forcing_params,t, qh=None):
    """
    Generates forcing based on specified wavenumber and time frequency
    F = A cos (B x + C t) + D cos(E y + F t)
    
    Dynamic flag ensures constant enstrophy injection \int (q * F) = A
    """
    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)

    # Create meshgrid for x, y
    X, Y = x[None,:],y[:,None]
    
    if forcing_params.dynamic:
        # Dynamic flag ensures constant enstrophy injection \int (q * F) = A
        w =  torch.tensor(-(torch.cos(forcing_params.B * X + forcing_params.C * t)) + (torch.cos(forcing_params.E * Y + forcing_params.F * t)))
        wh = to_spectral(w)
        wh_enstrophy = int_cross_sq(wh,qh, grid) / (grid.Lx * grid.Ly)
        eps = 1e-12
        wh = forcing_params.D*wh/(wh_enstrophy + torch.sign(wh_enstrophy) * eps)
    else:
        w =  forcing_params.A * (torch.cos(forcing_params.B * X + forcing_params.C * t)) + forcing_params.D * (torch.cos(forcing_params.E * Y + forcing_params.F * t))
        wh = to_spectral(w)
    
    return dealias(wh,spectral_derivative,1/3)

def sin_forcing(grid,spectral_derivative,forcing_params,t, qh=None):
    """
    Generates forcing based on specified wavenumber and time frequency
    F = A sin (B x + C t) + D sin(E y + F t)
    
    Dynamic flag ensures constant enstrophy injection \int (q * F) = A
    """
    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, grid.Lx, grid.Nx,device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny,device=grid.device)

    # Create meshgrid for x, y
    X, Y = x[None,:],y[:,None]
    
    if forcing_params.dynamic:
        # Dynamic flag ensures constant enstrophy injection \int (q * F) = A
        w =  torch.tensor(-(torch.sin(forcing_params.B * X + forcing_params.C * t)) + (torch.sin(forcing_params.E * Y + forcing_params.F * t)))
        wh = to_spectral(w)
        wh_enstrophy = int_cross_sq(wh,qh, grid) / (grid.Lx * grid.Ly)
        eps = 1e-12
        wh = forcing_params.D*wh/(wh_enstrophy + torch.sign(wh_enstrophy) * eps)
    else:
        w =  forcing_params.A * (torch.sin(forcing_params.B * X + forcing_params.C * t)) + forcing_params.D * (torch.sin(forcing_params.E * Y + forcing_params.F * t))
        wh = to_spectral(w)
    
    return dealias(wh,spectral_derivative,1/3)

def piecewise_forcing(grid, spectral_derivative, forcing_params, t=None, qh=None):
    """
    Generates the piecewise forcing

                  -tau0 * 2pi/(0.9 Ly) * sin(pi y / g(x))           , y < g(x)
    F(x,y) =
                   tau0 * 2pi/(0.9 Ly) * sin(pi(y-g(x))/(Ly-g(x)))  , y >= g(x)

    where

        g(x) = Lx/2 + 0.2(x - Lx/2)
    """

    x = torch.linspace(0, grid.Lx, grid.Nx, device=grid.device)
    y = torch.linspace(0, grid.Ly, grid.Ny, device=grid.device)

    X, Y = x[None, :], y[:, None]

    # g(x)
    gx = grid.Lx / 2 + 0.2 * (X - grid.Lx / 2)

    eps = 1e-12
    gx = torch.clamp(gx, eps, grid.Ly - eps)

    amp = forcing_params.tau0 * (2 * np.pi) / (0.9 * grid.Ly)

    w = torch.where(
        Y < gx,
        -amp * torch.sin(np.pi * Y / gx),
        amp * torch.sin(np.pi * (Y - gx) / (grid.Ly - gx))
    )

    if forcing_params.dynamic:
        wh = to_spectral(w)
        wh_enstrophy = int_cross_sq(wh, qh, grid) / (grid.Lx * grid.Ly)

        wh = wh / (wh_enstrophy + torch.sign(wh_enstrophy) * eps)
    else:
        wh = to_spectral(w)

    return dealias(wh, spectral_derivative, 1/3)