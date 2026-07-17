import torch
import math
from Operators.spectral_conversion import to_physical, to_spectral, dealias
from Operators.operators import SpectralDerivatives
from Grid.grid import Grid

def create_circular_mask(grid, r,x_center=0.5, y_center=0.5, tolerance=1e-3):
    """
    This is a circle grid at the center
    """
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, steps=Nx, device= grid.device)
    y = torch.linspace(0, Ly, steps=Ny, device= grid.device)

    # Create meshgrid for x, y
    X, Y = x[None,:],y[:,None]

    # Find the center of the domain
    if x_center is None:
        x_center = 0.5
    if y_center is None:
        y_center=0.5
    
    x_center = Lx * x_center
    y_center = Ly * y_center

    # Compute the distance of each point from the center
    distance = torch.sqrt((X - x_center)**2 + (Y - y_center)**2)

    # Create the mask: inside the circle (distance < r) is 1
    mask = torch.zeros_like(distance, dtype=torch.float32,device=grid.device)
    mask[distance < r] = 1  # Inside the circle
    mask[torch.abs(distance - r) < tolerance] = 0.5  # Boundary (within tolerance)
    
    return mask 

def create_circular_mask_v2(grid, r,x_center=0.5, y_center=0.5, tolerance=1e-3):
    """
    This is a circle grid at the center (v2 indicates smoothed masks with a Gaussian filter)
    """
    # Use grid object for domain size and number of grid points
    
    Lx = 8*math.pi 
    Ly = 8*math.pi 
    Nx = 1024
    Ny = 1024
    
    grid_mask = Grid(Lx,Ly,Nx,Ny)
    spec_deriv_mask = SpectralDerivatives(grid_mask)

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, steps=Nx, device= grid.device)
    y = torch.linspace(0, Ly, steps=Ny, device= grid.device)

    # Create meshgrid for x, y
    X, Y = x[None,:],y[:,None]

    # Find the center of the domain
    if x_center is None:
        x_center = 0.5
    if y_center is None:
        y_center=0.5
    
    x_center = Lx * x_center
    y_center = Ly * y_center

    # Compute the distance of each point from the center
    distance = torch.sqrt((X - x_center)**2 + (Y - y_center)**2)

    # Create the mask: inside the circle (distance < r) is 1
    mask = torch.zeros_like(distance, dtype=torch.float32,device=grid.device)
    mask[distance < r] = 1  # Inside the circle
    mask[torch.abs(distance - r) < tolerance] = 0.5  # Boundary (within tolerance)
    
    mask_h=to_spectral(mask)
    G_gaussian = torch.exp(-4*spec_deriv_mask.krsq*((4*grid_mask.dx)**2)/24)
    G_cutoffx =  spec_deriv_mask.kr<(4*torch.pi/(4*grid_mask.dx))
    G_cutoffy =  spec_deriv_mask.ky<(4*torch.pi/(4*grid_mask.dx))
    G_test = G_cutoffy* G_cutoffx*G_gaussian
    mask_h = G_test*mask_h
    mask=to_physical(mask_h)
    scale = int(Nx/grid.Nx)
       
    return mask[::scale,::scale] 

def create_cape_mask_v1(grid, r, y_top, x_center, y_center, y_width,x_left, x_right, x_scale=0.5,y_scale=2, tolerance=1e-3):
    """
    This is a single cape
    """
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny
    
    if x_scale is None:
        x_scale = 0.5
    if y_scale is None:
        y_scale=2
    
    y_top = y_top*Ly
    x_center = x_center*Lx
    y_center = y_center*Ly

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, steps=Nx, device= grid.device)
    y = torch.linspace(0, Ly, steps=Ny, device= grid.device)

    # Create meshgrid for x, y
    X = x.unsqueeze(0).expand(Ny, Nx)
    Y = y.unsqueeze(1).expand(Ny, Nx)

    # Define cape function
    cape_fun = y_center+y_scale*torch.exp(-((X-x_center)/x_scale)**2)
    
    # Create the mask:
    mask = torch.zeros_like(cape_fun, dtype=torch.float32,device=grid.device)
    mask[(Y<cape_fun)] = 1 # Inside the land and cape
    mask[(torch.abs(Y-cape_fun)< tolerance)] = 0.5  # Boundary (within tolerance)
    
    # If cape should not touch walls
    if y_width is not None:
        mask[Y < (y_center - y_width*Ly)] = 0
    if (x_left is not None) and (x_right is not None):
        mask[(X < (x_left*Lx)) | (X > (x_right*Lx))] = 0

    return mask

def create_cape_mask_v2(grid, r, y_top, x_center, y_center, y_width,x_left, x_right, x_scale=0.5,y_scale=2, tolerance=1e-3):
    """
    This is a single cape (v2 indicates smoothed masks with a Gaussian filter)
    """
    # Use grid object for domain size and number of grid points
    Lx = 8*math.pi 
    Ly = 8*math.pi 
    Nx = 1024
    Ny = 1024
    
    grid_mask = Grid(Lx,Ly,Nx,Ny)
    spec_deriv_mask = SpectralDerivatives(grid_mask)
    
    if x_scale is None:
        x_scale = 0.5
    if y_scale is None:
        y_scale=2
    
    y_top = y_top*Ly
    x_center = x_center*Lx
    y_center = y_center*Ly

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, steps=Nx, device= grid.device)
    y = torch.linspace(0, Ly, steps=Ny, device= grid.device)

    # Create meshgrid for x, y
    X = x.unsqueeze(0).expand(Ny, Nx)
    Y = y.unsqueeze(1).expand(Ny, Nx)

    # Define cape function
    cape_fun = y_center+y_scale*torch.exp(-((X-x_center)/x_scale)**2)
    
    # Create the mask:
    mask = torch.zeros_like(cape_fun, dtype=torch.float32,device=grid.device)
    mask[(Y<cape_fun)] = 1 # Inside the land and cape
    mask[(torch.abs(Y-cape_fun)< tolerance)] = 0.5  # Boundary (within tolerance)
    
    # If cape should not touch walls
    if y_width is not None:
        mask[Y < (y_center - y_width*Ly)] = 0
    if (x_left is not None) and (x_right is not None):
        mask[(X < (x_left*Lx)) | (X > (x_right*Lx))] = 0
        
    mask_h=to_spectral(mask)
    G_gaussian = torch.exp(-4*spec_deriv_mask.krsq*((4*grid_mask.dx)**2)/24)
    G_cutoffx =  spec_deriv_mask.kr<(4*torch.pi/(4*grid_mask.dx))
    G_cutoffy =  spec_deriv_mask.ky<(4*torch.pi/(4*grid_mask.dx))
    G_test = G_cutoffy* G_cutoffx*G_gaussian
    mask_h = G_test*mask_h
    mask=to_physical(mask_h)
    scale = int(Nx/grid.Nx)
       
    return mask[::scale,::scale] 

# New version of sponge with linear ramp
def create_sponge(grid, x_right,  x_left, y_top, y_bottom, tolerance=1e-3):
    """
    This is a sponge at specific boundaries with linear ramp
    """
    # Use grid object for domain size and number of grid points
    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny  
    
    if y_top is not None:
        y_top = y_top * Ly
    if x_left is not None:
        x_left = x_left * Lx
    if y_bottom is not None:
        y_bottom = y_bottom * Ly
    if x_right is not None:
        x_right = x_right * Lx

    # Create a grid of coordinates (x, y)
    x = torch.linspace(0, Lx, steps=Nx, device= grid.device)
    y = torch.linspace(0, Ly, steps=Ny, device= grid.device)

    # Create meshgrid for x, y
    X = x.unsqueeze(0).expand(Ny, Nx)
    Y = y.unsqueeze(1).expand(Ny, Nx)
    
    mask = torch.zeros_like(X, dtype=torch.float32,device=grid.device)
    
    ramps = []

    # 1) Vertical strip between y_top<â†’y_bottom
    if (y_top is not None) and (y_bottom is not None) and (y_top < y_bottom):
        between = (Y > y_top) & (Y < y_bottom)
        ramp = torch.zeros_like(Y)
        ramp[between] = (Y[between] - y_top) / (y_bottom - y_top)
        ramps.append(ramp)

    else:
        # Top boundary (y_topâ†’Ly)
        if y_top is not None:
            ramp = (Y - y_top) / (Ly - y_top)
            ramps.append( ramp.clamp(min=0.0) )

        # Bottom boundary (0â†’y_bottom)
        if y_bottom is not None:
            ramp = (y_bottom - Y) / y_bottom
            ramps.append( ramp.clamp(min=0.0) )

    # 2) Horizontal strip between x_right<â†’x_left
    if (x_left is not None) and (x_right is not None) and (x_right < x_left):        
        between = (X > x_right) & (X < x_left)
        ramp = torch.zeros_like(X)
        ramp[between] = (X[between] - x_right) / (x_left - x_right)
        # Y limits on sponge
#         outside = (Y < 0.1*Ly) | (Y > 0.9*Ly)
#         ramp[outside] = 0.0
        ramps.append(ramp.clamp(0.0, 1.0))

    else:
        # Left boundary (0â†’x_left)
        if x_left is not None:
            ramp = (x_left - X) / x_left
            ramp = ramp.clamp(min=0.0)
            # Y limits on sponge
#             outside = (Y < 0.1*Ly) | (Y > 0.9*Ly)
#             ramp[outside] = 0.0
            ramps.append(ramp)

        # Right boundary (x_rightâ†’Lx)
        if x_right is not None:
            ramp = (X - x_right) / (Lx - x_right)
            ramp = ramp.clamp(min=0.0)
            # Y limits on sponge
#             outside = (Y < 0.1*Ly) | (Y > 0.9*Ly)
#             ramp[outside] = 0.0
            ramps.append(ramp)

    # If no boundaries were specified, just return zeros
    if not ramps:
        return torch.zeros_like(X)

    # Stack all ramps and take the max at every point
    all_ramps = torch.stack(ramps, dim=0)    # shape (num_boundaries, Ny, Nx)
    mask = all_ramps.max(dim=0).values       # shape (Ny, Nx)

    # Finally clamp to [0,1] to guard against any tiny numerical overshoot
    return mask.clamp(0.0, 1.0)

def create_boundary_mask(grid, width=0.0, tolerance=1e-3):
    """
    Binary boundary mask.

    Boundary = 1
    Interior = 0

    width is given as a fraction of the domain length.
    """

    Lx = grid.Lx
    Ly = grid.Ly
    Nx = grid.Nx
    Ny = grid.Ny

    x = torch.linspace(0, Lx, Nx, device=grid.device)
    y = torch.linspace(0, Ly, Ny, device=grid.device)

    X = x[None, :]
    Y = y[:, None]

    # Create meshgrid for x, y
    X = x.unsqueeze(0).expand(Ny, Nx)
    Y = y.unsqueeze(1).expand(Ny, Nx)

    wx = width * Lx
    wy = width * Ly

    mask = torch.zeros_like(X, dtype=torch.float32,device=grid.device)

    mask[X < wx] = 1          # left
    mask[X > Lx - wx] = 1     # right
    mask[Y < wy] = 1          # bottom
    mask[Y > Ly - wy] = 1     # top

    mask[(X >= wx) & (X < wx + tolerance)] = 0.5
    mask[(X <= Lx - wx) & (X > Lx - wx - tolerance)] = 0.5
    mask[(Y >= wy) & (Y < wy + tolerance)] = 0.5
    mask[(Y <= Ly - wy) & (Y > Ly - wy - tolerance)] = 0.5

    return mask