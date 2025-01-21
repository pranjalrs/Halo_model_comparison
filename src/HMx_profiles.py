import numpy as np
import scipy.integrate

import astropy.units as u
import astropy.cosmology.units as cu
import astropy.constants as const

# Convention
# r: physical radial distance
# x: distance in r/Rvir
# y: distance in r/Rs where Rs = Rvir/c

def get_Pe_profile(M, z, params, x_bins=None):
    c_M = get_concentration(M, z, params)
    rvirial = get_rvirial(M, z, params)

    rho_bnd = get_rho_gas_profile(M, z, params, x_bins)[0]
    Temp_g = get_Temp_g(M, z, params, x_bins)
    P_e = rho_bnd * const.k_B*Temp_g/const.m_p
    
    return P_e, x_bins


def get_rho_dm_profile(M, z, params, x_bins=None):
    if x_bins is None:
        x_bins = np.logspace(np.log10(0.1), np.log10(1), 200)

    c_M = get_concentration(M, z, params)
    fcdm = 1 - params['omega_b']/params['omega_m']
    Mcdm = M*fcdm
    
    rvirial = get_rvirial(M, z, params)
    vol = 4/3*np.pi*rvirial**3
    rho_halo = Mcdm/vol
    A_NFW = np.log(1 + c_M) - c_M/(1+c_M)
    
    denom = 3*A_NFW*x_bins*(1/c_M + x_bins)**2
    rho_cdm = rho_halo/denom


    return rho_cdm, x_bins

def get_rho_gas_profile(M, z, params, x_bins=None):
    if x_bins is None:
        x_bins = np.logspace(np.log10(0.1), np.log10(1), 200)

    rvirial = get_rvirial(M, z, params)
    c_M = get_concentration(M, z, params)
    rho_bnd = _get_rho_bnd(x_bins*rvirial, rvirial, c_M, params)

    norm = get_norm(_get_rho_bnd, r_virial=rvirial, c_M=c_M, params=params)

    return rho_bnd*get_f_bnd(M, params)*M / norm, x_bins


def get_Temp_g(M, z, params, x_bins):
    '''Gas temperature
    Eq. 38
    '''
    alpha = params['alpha']
    f_H = 0.76
    mu_p = 4/(3 + 5*f_H)
    mu_e = 2/(1 + f_H)
    
    r_virial = get_rvirial(M, z, params)
    c_M = get_concentration(M, z, params)
    #     Tv = G * m_p * mu_p /(a * rvirial) /(3/2 * kB) * M
    T_virial = alpha*(const.G*const.m_p*mu_p*(1+z)/r_virial/(3/2*const.k_B)*M).to(u.K)
    

    r_s = r_virial/c_M
    r_bins = x_bins * r_virial
    y = (r_bins/r_s).decompose()

    f_r = np.log(1 + y)/y
    return T_virial * (f_r)/mu_e




def _get_rho_bnd(r, r_virial, c_M, params):
    gamma = params['gamma']
    Rs = r_virial/c_M
    y = r/Rs
    return np.power((np.log(1+y) / y ), 1/(gamma-1) )


def get_norm(profile, r_virial, c_M, params):
    r_unit = r_virial.unit
    integrand = lambda r: 4*np.pi*r**2*profile(r*u.Mpc/cu.littleh, r_virial=r_virial, c_M=c_M, params=params)
    rrange = np.linspace(1e-6, r_virial.value, 2000)  # Integration range 0, 1Rvir
    y = integrand(rrange)
    return scipy.integrate.simpson(y, rrange)*r_unit**3


def get_delta_v(z, params):
    '''Eq. 22
    '''
    omega_at_z = params['omega_m']*(1+z)**3 / (params['omega_m']*(1+z)**3 + (1- params['omega_m']))  # Omega0*(1+z)^3/E(z)^2
    return 1/omega_at_z * (18*np.pi**2 -82*(1-omega_at_z) - 39*(1-omega_at_z)**2)


def get_rvirial(M, z, params):
    """_summary_

    Parameters
    ----------
    M : float
        Halo mass in Mass units/h
    z : float, optional
        redshift, by default 0.0

    Returns
    -------
    float
        virial radius in Mpc/h
    """
    delta_v = get_delta_v(z, params)
    rho_crit = 2.7554e11 * u.Msun/u.Mpc**3 * cu.littleh**2  # In Msun * h**2 /Mpc**3
    rho_m = params['omega_m'] * rho_crit

    return ((M/ (4/3*np.pi * delta_v * rho_m))**(1/3)).to(u.Mpc/cu.littleh)#*self.h**(2/3)  #in Mpc


def get_f_bnd(M, params):
    """Eq. 25
    """
    return params['omega_b']/params['omega_m'] * (M/params['M0'])**params['beta']/(1 + (M/params['M0'])**params['beta'])


def get_concentration(M, z, params):
    '''Eq. 33
    M should be in Msun/h
    '''
    ## Concenetraion-Mass relation from Duffy et. al. 2008
    MSCALE = 2e12*u.Msun/cu.littleh
    c_M = 7.85 * (M/MSCALE)**(-0.081) * (1+z)**(-0.71)


    eps1 = params['eps1_0'] + params['eps1_1']*z
    eps2 = params['eps2_0'] + params['eps2_1']*z

    if eps1<= -1: 
        raise ValueError("eps1<-1 concentration for low mass halos is negative!")

    if eps2<= -1: 
        raise ValueError("eps2<-1 concentration for high mass halos is negative!")

    c_M_modified = c_M * (1 + eps1 + (eps2-eps1) * get_f_bnd(M, params)/ (params['omega_b']/params['omega_m']))
    c_M_modified = c_M_modified#*self.HMcode_rescale_A

    return c_M_modified
