import numpy as np

import pyccl as ccl
import BaryonForge as bfg

#Define relevant physical constants
Msun_to_Kg = ccl.physical_constants.SOLAR_MASS
Mpc_to_m   = ccl.physical_constants.MPC_TO_METER
G          = ccl.physical_constants.GNEWT / Mpc_to_m**3 * Msun_to_Kg
m_to_cm    = 1e2
kb_cgs     = ccl.physical_constants.KBOLTZ * 1e7
K_to_kev   = ccl.physical_constants.KBOLTZ / ccl.physical_constants.EV_IN_J * 1e-3

#Just define some useful conversions/constants
sigma_T = 6.652458e-29 / Mpc_to_m**2
m_e     = 9.10938e-31 / Msun_to_Kg
m_p     = 1.67262e-27 / Msun_to_Kg
c       = 2.99792458e8 / Mpc_to_m

bpar_S19 = dict(gamma = 2, delta = 3,
				M_c = 1e14, mu_beta = 0.2,
				theta_co = 0.1, M_theta_co = 13, mu_theta_co = -0.2, zeta_theta_co = 0.2,
				theta_ej = 4, M_theta_ej = 13, mu_theta_ej = -0.2, zeta_theta_ej = 0.2,
				eta = 0.3, eta_delta = 0.3, tau = -1.5, tau_delta = 0, #Must use tau here since we go down to low mass
				A = 0.09/2, M1 = 10**(11.5), epsilon_h = 0.015,
				alpha_nt = 0.18, gamma_nt = 0.8, nu_nt=0.,
				a = 0.3, n = 2, epsilon = 4, p = 0.3, q = 0.707)

bpar_A20 = dict(alpha_g = 2, epsilon_h = 0.015, M1_0 = 1e12,
				alpha_fsat = 1, M1_fsat = 1, delta_fsat = 1, gamma_fsat = 1, eps_fsat = 1,
				M_c = 1e13, eta = 0.5, mu = 0.31, beta = 0.35, epsilon_hydro = np.sqrt(5),
				M_inn = 1e13, M_r = 10**(13.5), beta_r = 2, theta_inn = 0.2, theta_out = 2,
				theta_rg = 0.3, sigma_rg = 0.1, a = 0.3, n = 2, p = 0.3, q = 0.707,
				A_nt = 0.495, alpha_nt = 0.08,
				mean_molecular_weight = 0.59)


def get_param_grid(param, p_dict, ngrid):
	this_prior = p_dict[param]['prior']
	this_value = p_dict[param]['initial_value']

	dx = (this_prior[1] - this_prior[0]) / ngrid

	assert ngrid % 2 != 0, 'ngrid must be odd'
	# Check if min and max possible values are within prior
	nstep_down = int((this_value - this_prior[0]) // dx)
	nstep_up = int((this_prior[1] - this_value) // dx)

	if nstep_down + nstep_up + 1 > ngrid:
		if nstep_down > nstep_up:
			nstep_down -= 1
		else:
			nstep_up -= 1

	elif nstep_down + nstep_up + 1 < ngrid:
		if nstep_down < nstep_up:
			nstep_down += 1
		else:
			nstep_up += 1

	param_grid = [this_value - dx * i for i in range(nstep_down, 0, -1)]
	param_grid += [this_value]
	param_grid += [this_value + dx * i for i in range(1, nstep_up + 1)]

	assert len(param_grid) == ngrid

	return param_grid


def get_param_dict(param_values, param_names, param_dict, halo_model):
	if halo_model == 'Arico20':
		par = bpar_A20.copy()

	elif halo_model == 'Mead20':
		par = bfg.Profiles.Mead20.Params_TAGN_7p8_MPr.copy()

	elif halo_model == 'Schneider19':
		par = bpar_S19.copy()

	assert (param_values is not None and param_names is not None) or param_dict is not None, 'Please provide either param_values and param_names or param_dict'

	if param_dict is not None:
		par.update(param_dict)

	else:

		for key, val in zip(np.atleast_1d(param_names), np.atleast_1d(param_values)):
			if key == 'alpha_sat':
				par['M1_fsat'] = val
				par['eps_fsat'] = val
				par['alpha_fsat'] = val
				par['delta_fsat'] = val
				par['gamma_fsat'] = val

			elif key.startswith('log'):
				if key.split('log')[1] not in par.keys(): raise ValueError(f'Invalid parameter {key}!')
				par[key.split('log')[1]] = 10**val

			else:
				if key not in par.keys(): raise ValueError(f'Invalid parameter {key}!')
				par[key] = val


	return par


def get_halo_model_Mead20(bfg_dict, par):
			#Define profiles. Normalize to convert density --> overdensity
	DMB = bfg.Profiles.Mead20.DarkMatterBaryon(**par, mass_def = bfg_dict['mdef']) / bfg_dict['rho']
	PRS = bfg.Profiles.Mead20.Pressure(**par, mass_def=bfg_dict['mdef'])
	GAS = bfg.Profiles.Mead20.Gas(**par, mass_def=bfg_dict['mdef'])
	ElectronDensity = bfg.Profiles.GasNumberDensity(gas = GAS, mean_molecular_weight = 1.15, mass_def=bfg_dict['mdef']) #simple constant rescaling of gas density --> number density in cgs
	ElectronDensity_sq = ElectronDensity**2


	def get_fgas(mass):
		fbnd, fej = DMB._get_gas_frac(mass, 1, bfg_dict['cosmo'])
		return (fbnd + fej) * mass


	mean_fgas = bfg_dict['HMC'].integrate_over_massfunc(get_fgas, bfg_dict['cosmo'], a=1)
	mean_mass = bfg_dict['HMC'].integrate_over_massfunc(lambda mass: mass, bfg_dict['cosmo'], a=1)
	mean_fgas = mean_fgas/mean_mass

	mean_electron_density = bfg_dict['rho']/(m_p*(Mpc_to_m * m_to_cm)**3)/1.15 * mean_fgas
	delta_ElectronDensity = ElectronDensity / mean_electron_density

	for p in [DMB, PRS, GAS, ElectronDensity, ElectronDensity_sq, delta_ElectronDensity]:
		p.update_precision_fftlog(**bfg_dict['fft_precision'])

	return DMB, PRS, GAS, ElectronDensity, ElectronDensity_sq, delta_ElectronDensity


def get_bfg_profile(param_value=None, param_name=None, comp=None, bfg_dict=None, fourier=False):
	par = get_param_dict(param_value, param_name, None, halo_model=bfg_dict['halo_model'])


	cosmo = bfg_dict['cosmo']
	HaloMass = bfg_dict['HaloMass']
	HaloRadius = bfg_dict['HaloRadius']
	a = bfg_dict['a']

	if bfg_dict['halo_model'] == 'Mead20':
		GAS = bfg.Profiles.Mead20.Gas(**par, mass_def=bfg_dict['mdef'])
		STAR = bfg.Profiles.Mead20.Stars(**par, mass_def=bfg_dict['mdef'])


	elif bfg_dict['halo_model'] == 'Arico20':
		GAS = bfg.Profiles.Arico20.Gas(**par, r_min_int=1e-5, r_max_int=1e2, r_steps=500)
		STAR = bfg.Profiles.Arico20.Stars(**par, r_min_int=1e-6, r_max_int=10, r_steps=500)

	elif bfg_dict['halo_model'] == 'Schneider19':
		GAS = bfg.Profiles.Schneider19.Gas(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)
		STAR = bfg.Profiles.Schneider19.Stars(**par, r_min_int = 1e-6, r_max_int = 5, r_steps = 500)


	if comp == 'gas':
		prof = []
		for M, r in zip(HaloMass, HaloRadius):
			if fourier:
				prof.append(GAS.fourier(cosmo, M=M, k=2*np.pi/r, a=a))
			else:
				prof.append(GAS.real(cosmo, M=M, r=r, a=a))
		return prof

	elif comp == 'stars':
		prof = []
		for M, r in zip(HaloMass, HaloRadius):
			if fourier:
				prof.append(STAR.fourier(cosmo, M=M, k=2*np.pi/r, a=a))
			else:
				prof.append(STAR.real(cosmo, M=M, r=r, a=a))
		return prof