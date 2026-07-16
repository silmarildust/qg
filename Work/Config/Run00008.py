import math

class grid_params:
    Nx= 512
    Ny= 512
    Lx= 2*math.pi
    Ly= 2*math.pi
    
class time_params:
    dt= 5e-5
    T = 2000001 *dt
    save_int=8192 #(Frequency of .np saves and plots)
    
class pde_params:
    mu = 0.1  #(Linear drag)
    nu = 4e-5 #(Viscosity coefficient)
    B = 5 #(Beta plane)
    nv = 1 #(Hyperviscous order)
    penalty_coeff=1.25*time_params.dt  # (Brinkman penalty parameter)
    closure_option = None #No closure
    if closure_option is not None:
        c = 0.25 
        backscatter_flag = True
    if closure_option == 4:
        width = 2
        func_option = 1 # Leith
        A = [0,0] #Only A12 and A21. A11 and A22 are always on.
        alpha = 1 #Use None if no alpha, or else 0-1.
    
class ic_params:
    option = 1 # QG ICs
    seed= 27
    if option == 1:  #QG ICs
        energy= 0.0
        wavenumbers= [3.0, 5.0]
    if option == 2: #FPC ICs
        init_vel =-2
    if option == 3: #Closure ICs
        file_path = f'/gdata/projects/ml_scope/Turbulence/QG_V0003/Results/Run03517/fields_Run03517.npy'
        start_time = 101
        scale = 8
        Nx = 512
        Ny = 512
    
class mask_params:
    option = 3      # Boundary mask
    width = 0.025   # 2.5% of the domain
    # option = 1 
    # 1 is circular mask
    r = 0.0
    x_center = 0.25
    y_center =0.5
    tol = 1e-3
    
class forcing_params:
    option = 1 # 1 is sin forcing
    dynamic = True  # Constant enstrophy injection
    A=0 # A sin(x/B + Ct) + D sin(y/E + Ft)
    B=2
    C=0
    D=1.5
    E=2
    F=0

class sponge_params:
    option = 0 # 1 is boundary sponging
    x_left = 0.1
    x_right = 0.9
    y_top = None
    y_bottom = None
    tol = 1e-3

class params:
    grid = grid_params
    time = time_params
    pde = pde_params
    ic = ic_params
    mask = mask_params
    forcing = forcing_params
    sponge=sponge_params
    run_number = 3589