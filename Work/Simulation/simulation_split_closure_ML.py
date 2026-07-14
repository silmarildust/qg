import torch
import numpy as np
import math
from tqdm import tqdm
from Operators.spectral_conversion import to_physical, to_spectral, dealias, get_puv
from Time_marching.imex_schemes import backward_euler, CN2, AB2
from Operators.ML_SGS import get_SGS

class Simulation:
    def __init__(self, grid, pde_params, spectral_derivative, linear_operator, nonlinear_operator, initial_condition, time_params,
                 forcing_params, model, DNS_nonlinear_operator, field_std, mask=None, forcing=None, sponge = None):
        """
        Initialize the simulation parameters.
        """
        self.grid = grid
        self.device = grid.device
        self.params = pde_params
        self.spectral_derivative = spectral_derivative
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.initial_condition = initial_condition # In spectral space
        self.xi = mask # Xi is the mask variable
        self.forcing = forcing
        self.sponge = sponge
        self.dt = time_params.dt
        self.T = time_params.T
        self.save_interval=time_params.save_int
        self.forcing_params = forcing_params
        self.model = model # Diffusion model
        self.DNS_nonlinear_operator = DNS_nonlinear_operator # DNS (512x512) operators to compute SGS terms
        self.field_std = field_std # Normalizing factor to diffusion model
        self.Nx, self.Ny = grid.Nx, grid.Ny  # Grid resolution
        self.steps = int(self.T / self.dt)  # Number of time steps
        self.q_sol = torch.zeros([self.Nx, self.Ny, int(self.steps/self.save_interval)+1,4]) # Initialize for only the save times
        # Temporary assignments for spectral fields
        qh_temp,ph_temp, uh_temp,vh_temp  = self.initial_condition # Vorticity, Streamfunction, u velocity, v velocity
        # Permanent assigment for physical fields
        self.q_sol[:, :, 0,0] = to_physical(qh_temp)
        self.q_sol[:, :, 0,1] = to_physical(ph_temp) 
        self.q_sol[:, :, 0,2] = to_physical(uh_temp) 
        self.q_sol[:, :, 0,3] = to_physical(vh_temp) 
        self.t0=0.0
        # Collect upsampled vorticities
        self.q_DNS = []
        
    def time_step(self):
        """Perform the time-stepping loop."""
        dt = self.dt
        xi = self.xi
        if self.sponge is not None: 
            xi_sponge = self.sponge

        for it_count in range(self.steps - 1):    
            # Handle initial conditions
            if it_count == 0:
                q_sol_h_1 = to_spectral(self.q_sol[:, :, it_count,0].squeeze()).cuda() #Vorticity
                p_sol_h_1 = to_spectral(self.q_sol[:, :, it_count,1].squeeze()).cuda() #Streamfunction
                u_sol_h_1 = to_spectral(self.q_sol[:, :, it_count,2].squeeze()).cuda() # u velocity
                v_sol_h_1 = to_spectral(self.q_sol[:, :, it_count,3].squeeze()).cuda() # v velocity             
                u_sol_h_IC = u_sol_h_1[0,0]
                v_sol_h_IC = v_sol_h_1[0,0]    
            else:
                q_sol_h_1 = ans
                
                p_sol_h_1,u_sol_h_1,v_sol_h_1 = get_puv(q_sol_h_1,self.spectral_derivative)

                ## Re-set ICs in u and v (background flow)
                u_sol_h_1[0,0] = u_sol_h_IC
                v_sol_h_1[0,0] = v_sol_h_IC
                            
            # Initialize source term
            source = q_sol_h_1
              
            # Compute nonlinear operators
            nlo_jacobian_1=self.nonlinear_operator.jacobian_pq(q_sol_h_1, p_sol_h_1, u_sol_h_1, v_sol_h_1)
            nlo_brinkman_1=self.nonlinear_operator.brinkman_penalty(xi,q_sol_h_1, p_sol_h_1, u_sol_h_1, v_sol_h_1)    
                        
            # Compute closure terms
            q_DNS_1, source_closure_1 = get_SGS(self.model,self.DNS_nonlinear_operator,self.nonlinear_operator,self.field_std,
                                                q_sol_h_1,p_sol_h_1,u_sol_h_1,v_sol_h_1)
            
            if self.params.closure_option == 6:
                source_closure_1 += self.nonlinear_operator.smag_closure(self.params.c,q_sol_h_1, p_sol_h_1, u_sol_h_1, v_sol_h_1)
            elif self.params.closure_option == 7:
                source_closure_1 += self.nonlinear_operator.leith_closure(self.params.c,q_sol_h_1, p_sol_h_1, u_sol_h_1, v_sol_h_1)
            
            # Compute nonlinear terms
            if it_count == 0:
                source_jacobian = backward_euler(nlo_jacobian_1, dt)
                source_brinkman = backward_euler(nlo_brinkman_1, dt)
                if self.forcing:
                    if self.forcing_params.dynamic:
                        term_forcing1 = self.forcing(self.grid,self.spectral_derivative,self.forcing_params,self.t0+it_count*dt,q_sol_h_1)
                    else:
                        term_forcing1 = self.forcing(self.grid,self.spectral_derivative,self.forcing_params,self.t0+it_count*dt)
                    source_forcing = backward_euler(term_forcing1, dt)  
                if self.params.closure_option is not None:
                    source_closure = backward_euler(source_closure_1, dt)
            else:
                source_jacobian = AB2(nlo_jacobian_1, nlo_jacobian_2, dt)
                source_brinkman = AB2(nlo_brinkman_1, nlo_brinkman_2, dt)
                if self.forcing:
                    if self.forcing_params.dynamic:
                        term_forcing1 = self.forcing(self.grid,self.spectral_derivative,self.forcing_params,self.t0+it_count*dt,q_sol_h_1)
                    else:
                        term_forcing1 = self.forcing(self.grid,self.spectral_derivative,self.forcing_params,self.t0+it_count*dt)
                    source_forcing = AB2(term_forcing1,term_forcing2, dt)
                if self.params.closure_option is not None:
                    source_closure = AB2(source_closure_1, source_closure_2, dt)
                    
            # Compute linear terms
            source_lin, op_lin = CN2(self.linear_operator,q_sol_h_1,dt)         

            # Update the source term
            source = source + source_lin + source_jacobian + source_brinkman
            
            if self.forcing:
                source = source + source_forcing
                
            if self.params.closure_option is not None:
                source = source + source_closure

            # Apply the linear operator inversion
            operator = (1 - op_lin).cuda()
            ans = source / operator
                
            ### Spliting scheme (only sponge term now)
            if self.sponge is not None: 
                #Update all fields to temporary term
                source = ans
                q_sol_h_temp = ans
                p_sol_h_temp,u_sol_h_temp,v_sol_h_temp = get_puv(q_sol_h_temp,self.spectral_derivative)

                ## Re-set ICs in u and v (background flow)
                u_sol_h_temp[0,0] = u_sol_h_IC
                v_sol_h_temp[0,0] = v_sol_h_IC

                nlo_sponge_1=self.nonlinear_operator.sponge_penalty_v1(xi_sponge,q_sol_h_temp, p_sol_h_temp, u_sol_h_temp, v_sol_h_temp)

                if it_count == 0:
                    source_sponge = backward_euler(nlo_sponge_1, dt)
                else:
                    source_sponge = AB2(nlo_sponge_1, nlo_sponge_2, dt)            

                # Simply add sponge to temp vorticity
                source = source + source_sponge     
                ans = source       

            # Store input as term 2 for AB and CN
            nlo_jacobian_2=nlo_jacobian_1
            nlo_brinkman_2=nlo_brinkman_1
            if self.forcing:
                term_forcing2 = term_forcing1
            if self.sponge is not None:
                nlo_sponge_2 = nlo_sponge_1
            if self.params.closure_option is not None:
                source_closure_2 = source_closure_1

            # Convert back to physical space and store the result for every save interval
            if (it_count+1) % self.save_interval == 0:
                save_index = (it_count + 1) // self.save_interval
                qh_temp = ans
                ph_temp,uh_temp,vh_temp = get_puv(qh_temp,self.spectral_derivative)
                ## Re-set ICs in u and v (background flow)
                uh_temp[0,0] = u_sol_h_IC
                vh_temp[0,0] = v_sol_h_IC
                
                #Collect upsampled vorticities
                self.q_DNS.append(q_DNS_1.cpu())
                
                q_phys = to_physical(qh_temp)
                if torch.isnan(q_phys).any():
                    raise ValueError(f"NaN detected at time step {it_count+1}, save_index={save_index}")
                self.q_sol[:, :, save_index,0] = q_phys
                self.q_sol[:, :, save_index,1] = to_physical(ph_temp)
                self.q_sol[:, :, save_index,2] = to_physical(uh_temp)
                self.q_sol[:, :, save_index,3] = to_physical(vh_temp)               

    def run(self, return_dynamic = False):
        """Run the full simulation."""
        self.time_step()
        return self.q_sol  # Return the solution array
    
    def get_upsampled_vorticity(self):
        """Return the dynamic closure values array, or None if not used."""
        return np.array([q.detach().cpu().numpy() for q in self.q_DNS])