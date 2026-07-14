import torch
import numpy as np
import math

from Operators.spectral_conversion import to_physical, to_spectral, dealias

### Set up spectral derivatives (first and second derivatives)
# Conventions: first derivative: + 1j*k, second derivative: -k**2

### Set up spectral derivatives (first and second derivatives)
class SpectralDerivatives:
    def __init__(self, grid):
        self.grid = grid
        self.device = grid.device

        # Number of wavenumber components (half of real grid in x-direction)
        self.dk = int(grid.Nx / 2 + 1)

        ## Compute wavenumbers for first derivatives
        # Derivative in y
        self.ky = torch.reshape((torch.fft.fftfreq(grid.Ny, grid.Ly / (grid.Ny * 2 * math.pi))), 
            (grid.Ny, 1)
        ).to(self.device) 
        
        # Derivative in x
        self.kr = torch.reshape((torch.fft.rfftfreq(grid.Nx, grid.Lx / (grid.Nx * 2 * math.pi))), 
            (1, self.dk)
        ).to(self.device)

        # Squared wavenumbers (for second derivatives)
        self.krsq = self.kr**2 + self.ky**2  

        # Inverse squared wavenumbers (and handling zero division)
        self.irsq = 1.0/self.krsq
        self.irsq[0,0] = 0.0 #

    def to(self, device):
        """ Move spectral operator tensors to another device. """
        self.device = device
        self.ky = self.ky.to(device)
        self.kr = self.kr.to(device)
        self.krsq = self.krsq.to(device)
        self.irsq = self.irsq.to(device)

    def __repr__(self):
        return (f"SpectralDerivatives(Nx={self.grid.Nx}, Ny={self.grid.Ny}, dk={self.dk}, "
                f"Lx={self.grid.Lx:.4f}, Ly={self.grid.Ly:.4f}, device={self.device})")

### Set up linear operator
class LinearOperator:
    def __init__(self, spectral_derivative, params):
        self.spectral_derivative = spectral_derivative
        self.params = params
        self.device = spectral_derivative.device
        
        # Precompute the linear term and store it
        self.Lc = self.linear_term()

    def linear_term(self):
        """
        Computes the  linear terms and operators together: diffusion, bottom drag, Coriolis with beta term
        """
        # Extracting parameters from the params object
        nu = self.params.nu
        mu = self.params.mu
        B = self.params.B
        
        # Calculate the linear term 
        # first term is diffusion: nu del^2 omega
        # then bottom drag: - mu omega
        # then Coriolis with beta term: - beta d psi/ dx (where omega = del^2 psi)
        Lc = -nu * self.spectral_derivative.krsq - mu + 1j * torch.tensor(B).to(self.device) * self.spectral_derivative.kr * self.spectral_derivative.irsq
        return Lc

    def apply(self, input_field):
        """
        Multiplies the linear oeprator with the input_tensor.
        Assumes the input_field is on the same device as the LinearOperator.
        """
        # Ensure input_tensor is on the same device
        input_field = input_field.to(self.device)

        # Multiply the linear term by input_tensor
        out = self.Lc * input_field
        
        return out

    def __repr__(self):
        return (f"LinearOperator(nu={self.params.nu}, mu={self.params.mu}, "
                f"B={self.params.B}, device={self.device})")

### Set up variables used for closure operators
class SGSVars:
    def __init__(self, spectral_derivative):
        """
        Initialize all SGSvars
        Variables ending in h are in Fourier space
        """
        self.spectral_derivative = spectral_derivative
        self.device = spectral_derivative.device
        self.qh = None
        self.ph = None
        self.uh = None
        self.vh = None
        self.dqdx = None
        self.dqdy = None
        self.dqdxx = None
        self.dqdxy = None
        self.dqdyy = None
        self.dpdxx = None
        self.dpdxy = None
        self.dpdyy = None
        self.grad_q = None
        self.S_bar = None
        self.delta = None

    def update_vars(self, input_field_q,input_field_p,input_field_u,input_field_v, width=None, G_test=None):
        """
        Compute all the SGSvars for the given input variables 
        """
        # In spectral space
        self.qh= input_field_q.clone()
        self.ph= input_field_p.clone()
        self.uh= input_field_u.clone() 
        self.vh= input_field_v.clone() 
        self.delta = (self.spectral_derivative.grid.dx * self.spectral_derivative.grid.dy) ** 0.5

        if width is not None:
            self.delta = width*(self.spectral_derivative.grid.dx * self.spectral_derivative.grid.dy) ** 0.5
            qh_test = G_test*self.qh
            ph_test = G_test*self.ph
            uh_test = G_test*self.uh
            vh_test = G_test*self.vh
            
            self.qh= qh_test
            self.ph= ph_test
            self.uh= uh_test
            self.vh= vh_test
             
        self.dqdx = to_physical(dealias(1j* self.spectral_derivative.kr * self.qh,self.spectral_derivative,1/3))
        self.dqdy = to_physical(dealias(1j* self.spectral_derivative.ky * self.qh,self.spectral_derivative,1/3))
        self.dqdxx = to_physical(dealias(-(self.spectral_derivative.kr**2) * self.qh,self.spectral_derivative,1/3))
        self.dqdyy = to_physical(dealias(-(self.spectral_derivative.ky**2)  * self.qh,self.spectral_derivative,1/3))
        self.dqdxy = to_physical(dealias(- self.spectral_derivative.ky * self.spectral_derivative.kr* self.qh,self.spectral_derivative,
                                          1/3))
        self.dqdxxx = to_physical(dealias(-1j*(self.spectral_derivative.kr**3) * self.qh,self.spectral_derivative,
                                          1/3))
        self.dqdxxy = to_physical(dealias(-1j*(self.spectral_derivative.kr**2) * (self.spectral_derivative.ky)*
                                          self.qh,self.spectral_derivative, 1/3))
        self.dqdxyy = to_physical(dealias(-1j*(self.spectral_derivative.kr) * (self.spectral_derivative.ky**2)*
                                          self.qh,self.spectral_derivative, 1/3))
        self.dqdyyy = to_physical(dealias(-1j*(self.spectral_derivative.ky**3) * self.qh,self.spectral_derivative,
                                          1/3))
        
        self.dpdxx = to_physical(dealias(-(self.spectral_derivative.kr**2) * self.ph,self.spectral_derivative,1/3))
        self.dpdyy = to_physical(dealias(-(self.spectral_derivative.ky**2)  * self.ph,self.spectral_derivative,1/3))
        self.dpdxy = to_physical(dealias(- self.spectral_derivative.ky * self.spectral_derivative.kr* self.ph,self.spectral_derivative,
                                          1/3))
        self.dpdxxx = to_physical(dealias(-1j*(self.spectral_derivative.kr**3) * self.ph,self.spectral_derivative,
                                          1/3))
        self.dpdxxy = to_physical(dealias(-1j*(self.spectral_derivative.kr**2) * (self.spectral_derivative.ky)*
                                          self.ph,self.spectral_derivative, 1/3))
        self.dpdxyy = to_physical(dealias(-1j*(self.spectral_derivative.kr) * (self.spectral_derivative.ky**2)*
                                          self.ph,self.spectral_derivative, 1/3))
        self.dpdyyy = to_physical(dealias(-1j*(self.spectral_derivative.ky**3) * self.ph,self.spectral_derivative,
                                          1/3))
        
        self.grad_q = torch.sqrt(self.dqdx**2 +  self.dqdy**2)
        self.S_bar = torch.sqrt(4*(self.dpdxy**2) + (self.dpdxx-self.dpdyy)**2)   
        
### Set up nonlinear operator
class NonlinearOperator:
    def __init__(self, spectral_derivative, params):
        self.spectral_derivative = spectral_derivative
        self.params = params
        self.device = spectral_derivative.device

    def jacobian_pq(self, input_field_q,input_field_p,input_field_u,input_field_v):
        """
        Computes the Jacobian of q (vorticity) and p (streamfunction) in spectral space (h).
        """
        # In spectral space
        qh= input_field_q.clone()
        ph= input_field_p.clone()
        uh= input_field_u.clone() 
        vh= input_field_v.clone() 

        # In physical space
        q=to_physical(qh)
        u=to_physical(uh)
        v=to_physical(vh)

        # Calculate jacobian
        uq = u*q
        vq = v*q

        uqh=to_spectral(uq)
        vqh=to_spectral(vq)
        
        #[-d/dx (u*q) - d/dy (v*q)]
        out = - 1j*self.spectral_derivative.kr*uqh - 1j*self.spectral_derivative.ky*vqh

        return dealias(out,self.spectral_derivative,1/3)

    def brinkman_penalty(self,xi,input_field_q,input_field_p,input_field_u,input_field_v):
        """
        Computes the brinkman volume penalization for mask (xi) in spectral space (h).
         - curl (mask * velocity) / eta
        """
        penalty_coeff=self.params.penalty_coeff
        # In spectral space
        qh= input_field_q.clone()
        ph= input_field_p.clone()
        uh= input_field_u.clone() 
        vh= input_field_v.clone() 

        # In physical space
        u=to_physical(uh)
        v=to_physical(vh)

        # Calculate product
        u_xi = u*xi
        v_xi = v*xi

        u_xi_h=to_spectral(u_xi)
        v_xi_h=to_spectral(v_xi)
        
        #[-d/dx (xi*v) + d/dy (xi*u)]
        out = (1/penalty_coeff)*(-1j * self.spectral_derivative.kr * v_xi_h + 1j*self.spectral_derivative.ky*u_xi_h) 
        
        return dealias(out,self.spectral_derivative,1/3)
    
    def sponge_penalty_v1(self,xi,input_field_q,input_field_p,input_field_u,input_field_v):
        """
        Computes the sponge penalization for mask (xi) in spectral space (h).
        - mask * curl (velocity) / eta
        """
        penalty_coeff=self.params.penalty_coeff

        # In spectral space
        qh= input_field_q.clone()
        ph= input_field_p.clone()
        uh= input_field_u.clone() 
        vh= input_field_v.clone() 
                
        #[xi*(-d/dx (v) + d/dy (u))]
        #out = (1/penalty_coeff)*(1j * self.spectral_derivative.kr * vh - 1j*self.spectral_derivative.ky*uh) 
        out = (-1/penalty_coeff)*(qh) 
        
        #out_dealiased = dealias(out,self.spectral_derivative,1/3)
        out_physical = to_physical(out)
        
        out_physical = out_physical*xi
        out = to_spectral(out_physical)
        
        return dealias(out,self.spectral_derivative,1/3)
       
    def functional_closure(self,c,sgs_var,option, flux_flag=False):
        """
        Computes functional closures in spectral space (h).
            + (grad.(nu*grad(q))
            
        Note: c here is defined as the linear term to make it easy to handle in Dynamic version
        
        If option == 1:
            Leith closure
            nu = c * (delta_FF)^3 * sqrt[domega_dx^2 + domega_dy^2]
            c = C_L^3         
        If option == 2:
            Smagorinsky closure
            nu = c * (delta_FF)^2 * sqrt[4*d^2psi_dx_dy^2 + (d^2psi_dx^2 - d^2psi_dy^2)^2]
            c = C_S^2       
        """       
        # Use Delta_FF = 2 x Delta_LES for Gaussian filter
        if option ==1:
            # Compute leith kernel in physical space
            nu = c*sgs_var.grad_q*((2*sgs_var.delta)**3)
        elif option ==2:
            # Compute smag kernel in physical space
            nu = c*sgs_var.S_bar*((2*sgs_var.delta)**2)
        else:
            raise ValueError("option must be 1 (Leith) or 2 (Smagorinsky)")
        #Compute products (nu * grad(q))
        nu_dqdx = nu*sgs_var.dqdx
        nu_dqdy = nu*sgs_var.dqdy
        # Convert to spectral space
        nu_dqdxh=dealias(to_spectral(nu_dqdx), self.spectral_derivative, 1/3)
        nu_dqdyh=dealias(to_spectral(nu_dqdy), self.spectral_derivative, 1/3) 
        
        if flux_flag:
            # Return SGS fluxes
            return [nu_dqdxh,nu_dqdyh]
        else:
            # Take grad again (grad.(nu_l*grad(q))
            out = 1j*self.spectral_derivative.kr*nu_dqdxh + 1j*self.spectral_derivative.ky*nu_dqdyh
            return dealias(out,self.spectral_derivative,1/3)
    
    def clark_closure(self,sgs_var, flux_flag=False):
        """
        Computes the Clark closure (NGM4) in spectral space (h).
        NGM2: - (4*delta_FF^2/12) * [dqdxy*(dpdxx-dpdyy) + dpdxy*(dqdyy - dqdxx)]
        NGM4: - (16*delta_FF^4/288) * [-dqdxxx*dpdxxy + dqdxxy*(dpdxxx - 2dpdxyy) - dqdxyy*(dpdyyy - 2dpdxxy) + dqdyyy*dpdxyy]        
        """
        # Update all SGS variables    
        out_2= sgs_var.dqdxy*(sgs_var.dpdxx-sgs_var.dpdyy) + sgs_var.dpdxy*(sgs_var.dqdyy - sgs_var.dqdxx)   
        out_2= -out_2*4*sgs_var.delta**2/12
        
        out_4 = -sgs_var.dqdxxx*sgs_var.dpdxxy + sgs_var.dqdxxy*(sgs_var.dpdxxx - 2*sgs_var.dpdxyy) - sgs_var.dqdxyy*(sgs_var.dpdyyy - 2*sgs_var.dpdxxy)+ sgs_var.dqdyyy*sgs_var.dpdxyy
        out_4 = -out_4*16*sgs_var.delta**4/288
        
        out = out_2+out_4
        
        if flux_flag:
            # Return SGS fluxes
            out_2_x = (4*sgs_var.delta**2/12)*(sgs_var.dpdxy*sgs_var.dqdx + sgs_var.dpdyy*sgs_var.dqdy)
            out_2_y = -(4*sgs_var.delta**2/12)*(sgs_var.dpdxx*sgs_var.dqdx + sgs_var.dpdxy*sgs_var.dqdy)
            
            out_4_x = (16*sgs_var.delta**4/288)*(sgs_var.dpdxxy*sgs_var.dqdxx + 2*sgs_var.dpdxyy*sgs_var.dqdxy + sgs_var.dpdyyy*sgs_var.dqdyy)
            out_4_y = -(16*sgs_var.delta**4/288)*(sgs_var.dpdxxx*sgs_var.dqdxx + 2*sgs_var.dpdxxy*sgs_var.dqdxy + sgs_var.dpdxyy*sgs_var.dqdyy)
            
            out_x = out_2_x+out_4_x
            out_y = out_2_y+out_4_y
            
            return [dealias(to_spectral(out_x),self.spectral_derivative,1/3),dealias(to_spectral(out_y),self.spectral_derivative,1/3)]
        else:
            return dealias(to_spectral(out),self.spectral_derivative,1/3)
    
    def clark_closure_NGM2(self,sgs_var, flux_flag=False):
        """
        Computes the Clark closure (NGM2) in spectral space (h).
        - (4*delta_FF^2/12) * [dqdxy*(dpdxx-dpdyy) + dpdxy*(dqdyy - dqdxx)]
        """
        # Update all SGS variables    
        out= sgs_var.dqdxy*(sgs_var.dpdxx-sgs_var.dpdyy) + sgs_var.dpdxy*(sgs_var.dqdyy - sgs_var.dqdxx)   
        out= -out*4*sgs_var.delta**2/12
        
        if flux_flag:
            # Return SGS fluxes
            out_x = (4*sgs_var.delta**2/12)*(sgs_var.dpdxy*sgs_var.dqdx + sgs_var.dpdyy*sgs_var.dqdy)
            out_y = -(4*sgs_var.delta**2/12)*(sgs_var.dpdxx*sgs_var.dqdx + sgs_var.dpdxy*sgs_var.dqdy)
            return [dealias(to_spectral(out_x),self.spectral_derivative,1/3),dealias(to_spectral(out_y),self.spectral_derivative,1/3)]
        else:
            return dealias(to_spectral(out),self.spectral_derivative,1/3)

    
    def dynamic_closure(self,width,input_field_q,input_field_p,input_field_u,input_field_v, 
                         func_option, A, alpha, flux_flag=False):
        """
        Computes the 1 or 2 parameter Dynamic (Germano-Lilly variant) closures in spectral space (h).
        [1  A12; A21 1] [struct_model; func_model] = [b1; b2]
        
        If A12 and A21 = 1, 2-parameter Dynamic Mixed Model (DMM)
        If A12 = 0 with A21=1; Dynamic structural which is then corrected by Dynamic functional (SDMM-1)
        If A12 = 1 with A21=0; Dynamic functional which is then corrected by Dynamic structural (SDMM-2)
        If A12 and A21 = 0, Alpha model [alpha * functional + (1-alpha) * structural]
        """
        # Elements of A
        A12 = A[0]
        A21 = A[1]
        
        # Define test filters (2 times larger than LES grid)
        G_gaussian = torch.exp(-4*self.spectral_derivative.krsq*((width*self.spectral_derivative.grid.dx)**2)/24)
        G_cutoffx =  self.spectral_derivative.kr.abs()<(torch.pi/(width*self.spectral_derivative.grid.dx))
        G_cutoffy =  self.spectral_derivative.ky.abs()<(torch.pi/(width*self.spectral_derivative.grid.dx))
        G_test = G_cutoffy* G_cutoffx * G_gaussian
        # Update all SGS variables
        sgs_var=SGSVars(self.spectral_derivative)
        sgs_var.update_vars(input_field_q,input_field_p,input_field_u,input_field_v)  
        
        # Compute L terms  (Pi for test filter)
        L_right = G_test*self.jacobian_pq(sgs_var.qh,sgs_var.ph,sgs_var.uh,sgs_var.vh)
        # Compute all test variables
        sgs_var_test=SGSVars(self.spectral_derivative)
        sgs_var_test.update_vars(input_field_q,input_field_p,input_field_u,input_field_v,width,G_test)  
        # Compute H terms
        L_left =  -self.jacobian_pq(sgs_var_test.qh,sgs_var_test.ph,sgs_var_test.uh,sgs_var_test.vh)
        L = L_left + L_right
        
        # Initiliaze all dynamic spatial averages
        LH = torch.tensor(0.0)
        LM = torch.tensor(0.0)
        H_square = torch.tensor(0.0)
        M_square = torch.tensor(0.0)
        HM = torch.tensor(0.0)
        eps = 1e-12
        
        # M for functional models
        M_left =  self.functional_closure(1,sgs_var_test,func_option)
        M_right = -G_test*self.functional_closure(1,sgs_var,func_option)
        M =  M_left + M_right
        LM = torch.clamp(torch.mean(to_physical(L)*to_physical(M)),min=0.0)
        M_square = torch.mean(to_physical(M)*to_physical(M))    

        # H for structural models
        H_left =  self.clark_closure(sgs_var_test)
        H_right = -G_test*self.clark_closure(sgs_var)
        H =  H_left + H_right
        LH = torch.clamp(torch.mean(to_physical(L)*to_physical(H)),min=0.0)
        H_square = torch.mean(to_physical(H)*to_physical(H)) 
        
        if torch.isnan(LH) or torch.isnan(LM) or torch.isnan(M_square) or torch.isnan(H_square):
            raise ValueError("NaN in dynamic averages (H or M).")
           
        # Always computed 
        HM = torch.mean(to_physical(H)*to_physical(M))    
        c_struct = (LH -  LM * HM*A12/(M_square+eps))/((H_square - (HM**2)*A12*A21/(M_square+eps))+eps)
        c_func = (LM -  LH * HM*A21/(H_square+eps))/((M_square - (HM**2)*A12*A21/(H_square+eps))+eps)
        
        if A12 == 0 and A21 == 0:
            # Scale coefficients according to alpha model
            c_func = alpha*c_func
            c_struct = (1-alpha)*c_struct
        
        if flux_flag: 
            flux_func = self.functional_closure(c_func, sgs_var, func_option, flux_flag=True)
            flux_struct = self.clark_closure(sgs_var, flux_flag=True)
            out = [
                flux_func[0] + c_struct * flux_struct[0],
                flux_func[1] + c_struct * flux_struct[1]]
            out = [dealias(out[0], self.spectral_derivative, 1/3),
                   dealias(out[1], self.spectral_derivative, 1/3)]
        else:
            out = self.functional_closure(c_func,sgs_var,func_option, flux_flag=False) + c_struct*self.clark_closure(sgs_var,
                                                                                                                     flux_flag=False)  
            out = dealias(out, self.spectral_derivative, 1/3)
        # Log value of c
        if func_option == 2:   # Smag
            c_f = math.copysign(abs(c_func) ** 0.5, c_func)
        elif func_option == 1: # Leith
            c_f = math.copysign(abs(c_func) ** (1/3), c_func)

        return out, [c_f, c_struct]
    
    def __repr__(self):
        return (f"NonlinearOperator(penalty_coeff={self.params.penalty_coeff}, "
            f"device={self.device})")
    