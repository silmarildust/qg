import torch
import numpy as np
import math
import torch.nn.functional as F

from Operators.spectral_conversion import to_physical, to_spectral, dealias, get_puv

def LES_downscale(x_in,grid,spec_deriv,scale):
    """ Move spectral operator tensors to another device. 
    x_in: spectral field
    
    x_out: physical field (downscaled)
    """
    qh=x_in.squeeze()
    G = torch.exp(-4*spec_deriv.krsq*((scale*grid.dx)**2)/24)
    qh_dr = G*qh
    G = spec_deriv.ky.abs()<(torch.pi/(scale*grid.dx))
    qh_dr = G*qh_dr
    G =  spec_deriv.kr.abs()<(torch.pi/(scale*grid.dx))
    q_dr = to_physical(G*qh_dr)  
    x_out = F.avg_pool2d(q_dr.unsqueeze(0).unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0).squeeze(0) 
    
    return x_out
    
def get_SGS(model,DNS_nonlinop,fDNS_nonlinop,field_std,input_field_q,input_field_p,input_field_u,input_field_v):
    """ Computes SGS term given DNS fields
    q_DNS: Upsampled DNS vorticity
    Pi = J(omega_bar,psi_bar) - bar(J(omega,psi))
    """
    scale = DNS_nonlinop.spectral_derivative.grid.Nx//fDNS_nonlinop.spectral_derivative.grid.Nx
    
    # Normalize before input and de-normalize output
    input_field_q_in = to_physical(input_field_q/field_std).repeat_interleave(scale,dim=0).repeat_interleave(scale,dim=1) 
    q_DNS = field_std*model(x_cond=input_field_q_in).squeeze()
    
    qh = to_spectral(q_DNS)
    ph, uh, vh = get_puv(qh, DNS_nonlinop.spectral_derivative)
    # Get q and p in fDNS space
    fq = LES_downscale(qh, DNS_nonlinop.spectral_derivative.grid,DNS_nonlinop.spectral_derivative,scale)
    fp = LES_downscale(ph, DNS_nonlinop.spectral_derivative.grid,DNS_nonlinop.spectral_derivative,scale)
    fu = LES_downscale(uh, DNS_nonlinop.spectral_derivative.grid,DNS_nonlinop.spectral_derivative,scale)
    fv = LES_downscale(vh, DNS_nonlinop.spectral_derivative.grid,DNS_nonlinop.spectral_derivative,scale)
        
    Pi_2 = LES_downscale(-DNS_nonlinop.jacobian_pq(qh,ph,uh,vh),
                         DNS_nonlinop.spectral_derivative.grid,DNS_nonlinop.spectral_derivative,scale)
    
    Pi_1 = to_physical(-fDNS_nonlinop.jacobian_pq(to_spectral(fq),to_spectral(fp),to_spectral(fu),to_spectral(fv)))
    
    Pi = Pi_1 - Pi_2
    
    return q_DNS, to_spectral(Pi)
  