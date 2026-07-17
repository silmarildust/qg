import math

class grid_params:
    Nx= 1024
    Ny= 1024
    Lx= 8*math.pi
    Ly= 8*math.pi
    
class time_params:
    dt= 1e-4
    T = 1000001 *dt
    save_int=5000 #(Frequency of .np saves and plots)
    
class pde_params:
    mu = 0 #(Linear drag)
    nu = 5e-3  #(Viscosity coefficient)
    B = 1  #(Beta plane)
    nv = 1 #(Hyperviscous order)
    penalty_coeff= 1.25*time_params.dt  # (Brinkman penalty parameter)
    sponge_coeff = 125*time_params.dt
    ## Added closure
    closure_option = None
    if closure_option is not None:
        c = 0.075 
    if closure_option == 3 or closure_option == 4:
        width = 2
    
class ic_params:
    option = 2 #FPC ICs
    energy= 0
    wavenumbers= [10.0, 32.0]
    seed= 86   
    if option == 2:
        init_vel =2
    
class mask_params:
    option = 2 # 2 is cape_v1
    r = math.pi/4
    tol = 1e-3
    
    if option == 2: # Params specific to cape mask
        y_top = 1
        x_center = 0.20
        y_center = 0.075
        x_scale=1
        y_scale=4
        y_width = 0.05
        x_left = 0.05
        x_right = 0.85
    
class forcing_params:
    option = 1 # 1 is cos forcing
    A=0 # A cos(B x+ Ct) + D (cos E y + Ft)
    B=4
    C=0
    D=0
    E=4
    F=0

class sponge_params:
    option = 1 # 1 is boundary sponging
    x_left = 0.98
    x_right = 0.90
    y_top = 0.75
    y_bottom = 0.95
    tol = 1e-3
            
class params:
    grid = grid_params
    time = time_params
    pde = pde_params
    ic = ic_params
    mask = mask_params
    forcing = forcing_params
    sponge=sponge_params
    run_number = 1