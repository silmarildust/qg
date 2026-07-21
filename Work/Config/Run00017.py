import math

class grid_params:
    Nx= 256
    Ny= 256
    Lx= 2*math.pi
    Ly= 2*math.pi
    
class time_params:
    dt= 1e-4
    T = 1000001 *dt
    save_int=5000 #(Frequency of .np saves and plots)
    
class pde_params:
    mu = 0.1  #(Linear drag)
    nu = 5e-4 #(Viscosity coefficient)
    B = 5 #(Beta plane)
    nv = 1 #(Hyperviscous order)
    penalty_coeff=1.25*time_params.dt  # (Brinkman penalty parameter)
    closure_option = None #Mixed model framework
    if closure_option is not None:
        c = 0.25 
    if closure_option == 4:
        width = 2
        func_option = 1 # Leith
        A = [0.5,0.5] #Only A12 and A21. A11 and A22 are always on.
        alpha = None #Use None if no alpha, or else 0-1.
    
class ic_params:
    option = 1 # QG ICs
    seed= 27
    if option == 1:  #QG ICs
        energy= 0.0
        wavenumbers= [3.0, 5.0]
    if option == 2: #FPC ICs
        init_vel =2
    if option == 3: #Closure ICs
        file_path = f'/gdata/projects/ml_scope/Turbulence/QG_V0003/Results/Run03591/fields_Run03591.npy'
        start_time = 101
        scale = 8
        Nx = 1024
        Ny = 1024
    
class mask_params:
    option = 3 # 3 is boundary mask
#     r = math.pi/5
#     x_center = 0.25
#     y_center = 0.5
    width = 0
    tol = 1e-3
    
class forcing_params:
    option = 1 # 1 is cos forcing
    dynamic = False  # Constant enstrophy injection
    A=0 # A cos(B x+ Ct) + D (cos E y + Ft)
    B=0
    C=0
    D=-1
    E=1/2
    F=0

class sponge_params:
    option = 1
    x_left = 0.1
    x_right = 0.9
    y_bottom = 0.1
    y_top = 0.9
    tol = 1e-3

class params:
    grid = grid_params
    time = time_params
    pde = pde_params
    ic = ic_params
    mask = mask_params
    forcing = forcing_params
    sponge=sponge_params
    run_number = 13