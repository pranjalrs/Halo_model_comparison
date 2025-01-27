import corner
import emcee
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from schwimmbad import MPIPool
import sys

import astropy.units as u
import pyccl as ccl
import BaryonForge as bfg

Pk_data = {}
config = {}
bfg_dict = {}

h = 0.704
bpar_S19 = dict(theta_ej = 4, theta_co = 0.1, M_c = 1e14/h, mu_beta = 0.4,
				eta = 0.3, eta_delta = 0.3, tau = -1.5, tau_delta = 0, #Must use tau here since we go down to low mass
				A = 0.09/2, M1 = 2.5e11/h, epsilon_h = 0.015,
				a = 0.3, n = 2, epsilon = 4, p = 0.3, q = 0.707, gamma = 2, delta = 7)

bpar_A20 = dict(alpha_g = 2, epsilon_h = 0.015, M1_0 = 2.2e11/h, alpha_sat = 1,
				M_c = 1.2e14/h, eta = 0.6, mu = 0.31, beta = 0.6, epsilon_hydro = np.sqrt(5),
				M_inn = 3.3e13/h, M_r = 1e16, beta_r = 2, theta_inn = 0.1, theta_out = 3,
				theta_rg = 0.3, sigma_rg = 0.1, a = 0.3, n = 2, p = 0.3, q = 0.707,
				A_nt = 0.495, alpha_nt = 0.1,
				mean_molecular_weight = 0.59)

def init_bfg_model(halo_model, cosmo, a):
	k = np.logspace(-2, 1., 100)
	a = a
	cosmo = cosmo

	rho  = ccl.rho_x(cosmo, 1, 'matter', is_comoving = True)
	fft_precision = dict(padding_lo_fftlog = 1e-8, padding_hi_fftlog = 1e8, n_per_decade = 100)

	if halo_model == 'Arico20':
		mdef = ccl.halos.MassDef(Delta="200", rho_type="critical")
		DMO = bfg.Profiles.Arico20.DarkMatter() / rho


	elif halo_model == 'Mead20':
		mdef = ccl.halos.MassDef(Delta="vir", rho_type="matter")
		DMO = bfg.Profiles.Mead20.DarkMatter(mass_def = mdef) / rho


	elif halo_model == 'Schneider19':
		mdef = ccl.halos.MassDef(Delta="200", rho_type="critical")
		T   = bfg.Profiles.misc.Truncation(epsilon = 100)
		DMO = bfg.Profiles.Schneider19.DarkMatter(**bpar_S19, mdef=mdef, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 50)*T/rho


	#We will use the built-in, CCL halo model calculation tools.
	hmf = ccl.halos.MassFuncSheth99(mass_def=mdef, mass_def_strict = False, use_delta_c_fit = True)
	hbf = ccl.halos.HaloBiasSheth99(mass_def=mdef, mass_def_strict = False, use_delta_c_fit = True)
	HMC  = ccl.halos.halo_model.HMCalculator(mass_function = hmf, halo_bias = hbf,
											mass_def = mdef,
											log10M_min = 8, log10M_max = np.log10(2e15), nM = 100)


	# Compute DMO power spectrum for response calculation
	DMO.update_precision_fftlog(**fft_precision)
	P_mm_dmo = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k, a, DMO)*cosmo.cosmo.params.h**3

	# Pressure units of data is eV/cm^3
	cgs_to_eV_cm3__factor = (u.erg/u.cm**3).to(u.eV/u.cm**3)


	bfg_meta_dict = {'halo_model': halo_model,
					 'k': k,
					 'a': a,
					 'cosmo': cosmo,
					 'rho': rho,
					 'fft_precision': fft_precision,
					 'mdef': mdef,
					 'HMC': HMC,
					 'P_mm_dmo': P_mm_dmo,
					 'cgs_to_eV_cm3__factor': cgs_to_eV_cm3__factor}

	return bfg_meta_dict


def get_bfg_Pk(param_values=None, fit_params=None, field=None, param_dict=None):
	par = get_param_dict(param_values, fit_params, param_dict)
	h = bfg_dict['cosmo'].cosmo.params.h

	if bfg_dict['halo_model'] == 'Mead20':
		#Define profiles. Normalize to convert density --> overdensity
		DMB = getattr(bfg.Profiles, bfg_dict['halo_model']).DarkMatterBaryon(**par, mass_def = bfg_dict['mdef']) / bfg_dict['rho']
		PRS = getattr(bfg.Profiles, bfg_dict['halo_model']).Pressure(**par, mass_def=bfg_dict['mdef'])
		GAS = getattr(bfg.Profiles, bfg_dict['halo_model']).Gas(**par, mass_def=bfg_dict['mdef'])
		GasDensity = bfg.Profiles.GasNumberDensity(gas = GAS, mean_molecular_weight = 1.15, mass_def=bfg_dict['mdef']) #simple constant rescaling of gas density --> number density in cgs
		ElectronDensity = GasDensity

	elif bfg_dict['halo_model'] == 'Arico20':
		DMO = bfg.Profiles.Arico20.DarkMatter()
		T   = bfg.Profiles.misc.Truncation(epsilon = 1)
		BND = bfg.Profiles.Arico20.BoundGas(**par, r_min_int = 1e-5, r_max_int = 1e2, r_steps = 500)
		EJG = bfg.Profiles.Arico20.EjectedGas(**par)
		RAG = bfg.Profiles.Arico20.ReaccretedGas(**par)
		GAS = BND * T + EJG + RAG
		STR = bfg.Profiles.Arico20.Stars(**par, r_min_int = 1e-6, r_max_int = 10, r_steps = 500)

		#In practice, we should be using the ModifiedDarkMatter in the CLM profile.
		#But we could use the standard NFW profile to speed things up a bit.
		CLM = bfg.Profiles.Arico20.CollisionlessMatter(**par, darkmatter = DMO, max_iter = 4, reltol = 5e-2, r_steps = 100)
		DMB = bfg.Profiles.Arico20.DarkMatterBaryon(gas = GAS, stars = STR, collisionlessmatter = CLM)/ bfg_dict['rho']
		PRS = bfg.Profiles.Arico20.Pressure(**par, gas = BND + EJG + RAG)

	elif bfg_dict['halo_model'] == 'Schneider19':
		#For Schneider19, the DarkMatterBaryon profile includes the TwoHalo term.
		#So let's make sure to remove that. The ``DarkMatterBaryon'' class is
		#still good one to use since it makes sure Gas + star + DM is normalized
		#to match the total mass from the DMO case
		T   = bfg.Profiles.misc.Truncation(epsilon = 100)
		DMO = bfg.Profiles.Schneider19.DarkMatter(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 50)
		GAS = bfg.Profiles.Schneider19.Gas(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 50)
		STR = bfg.Profiles.Schneider19.Stars(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 50)
		CLM = bfg.Profiles.Schneider19.CollisionlessMatter(**par, max_iter = 2, reltol = 5e-2, r_steps = 100)
		DMB = bfg.Profiles.Schneider19.DarkMatterBaryon(**par,
														gas = GAS, stars = STR,
														collisionlessmatter = CLM, darkmatter = DMO,
														twohalo = bfg.Profiles.misc.Zeros(),
														r_steps = 100)
		PRS = bfg.Profiles.Pressure(gas = GAS, darkmatterbaryon = DMB, **par, r_min_int = 1e-4, r_max_int = 1e2, r_steps = 50)

		DMB = DMB/ bfg_dict['rho']*T


	#Upgrade precision of all profiles.
	for p in [DMB, PRS]: p.update_precision_fftlog(**bfg_dict['fft_precision'])
		#Compute all power spectra. This routine has some functionality to mimic the
	#transition-regime modelling of Mead++ in HMCode.
	if field == 'm-m':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(bfg_dict['cosmo'], bfg_dict['HMC'], bfg_dict['k'], bfg_dict['a'], DMB)*h**3

	elif field == 'm-p':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(bfg_dict['cosmo'], bfg_dict['HMC'], bfg_dict['k'], bfg_dict['a'], DMB, prof2 = PRS)*h**5
		Pk = Pk*bfg_dict['cgs_to_eV_cm3__factor']

	elif field == 'ne-ne':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(bfg_dict['cosmo'], bfg_dict['HMC'], bfg_dict['k'], bfg_dict['a'], ElectronDensity)*h**3

	elif field == 'gas':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(bfg_dict['cosmo'], bfg_dict['HMC'], bfg_dict['k'], bfg_dict['a'], GAS)*h**3

	return Pk, bfg_dict['k']

def get_param_dict(param_values, param_names, param_dict):
	if bfg_dict['halo_model'] == 'Arico20':
		par = bpar_A20.copy()

	elif bfg_dict['halo_model'] == 'Mead20':
		par = bfg.Profiles.Mead20.Params_TAGN_7p8_MPr.copy()

	elif bfg_dict['halo_model'] == 'Schneider19':
		par = bpar_S19.copy()

	assert (param_values is not None and param_names is not None) or param_dict is not None, 'Please provide either param_values and param_names or param_dict'

	if param_dict is not None:
		par.update(param_dict)

	else:

		for key, val in zip(np.atleast_1d(param_names), np.atleast_1d(param_values)):
			if key.startswith('log'):
				if key.split('log')[1] not in par.keys(): raise ValueError(f'Invalid parameter {key}!')
				par[key.split('log')[1]] = 10**val

			else:
				if key not in par.keys(): raise ValueError(f'Invalid parameter {key}!')
				par[key] = val


	return par

def log_likelihood(param_values):
	if param_values is not None:
		for i, this_param in enumerate(config['fit_params']):
			prior = config['priors'][i]
			if not (prior[0] <= param_values[i] <= prior[1]):
				return -np.inf

	if not config['fit_response']:
		raise ValueError('Please fix h units ')
	response_denom_theory = bfg_dict['P_mm_dmo'] if config['fit_response'] else 1
	response_denom_data = Pk_data['dmo']['Pk'] if config['fit_response'] else 1

	log_likelihood = 0.

	for field in config['fields']:
		Pk_theory, k = get_bfg_Pk(param_values, config['fit_params'], field)
		Pk_sim = Pk_data[field]['Pk']/response_denom_data
		k_sim = Pk_data[field]['k']
		variance = Pk_data[field]['variance']/response_denom_data**2

		# Note that CCL uses Mpc units for k; need to convert to h/Mpc
		Pk_theory_interp = np.interp(k_sim, k/bfg_dict['cosmo'].cosmo.params.h, Pk_theory/response_denom_theory)

		log_likelihood += -0.5*np.sum((Pk_theory_interp - Pk_sim)**2/variance)

	return log_likelihood


def run_mcmc(config):
	burnin = config['mcmc']['burnin']
	nwalkers = config['mcmc']['nwalkers']
	nsteps = config['mcmc']['nsteps']
	ndim = config['ndim']
	save_dir = config['save_dir']

	initial_guess = np.array(config['initial_guess']) + 1e-3*np.random.randn(nwalkers, ndim)

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
	sampler.run_mcmc(initial_guess, nsteps, progress=True)
	walkers = sampler.get_chain(flat=False)
	np.save(f'{save_dir}/samples.npy', walkers)
	flat_chain = sampler.get_chain(discard=int(burnin*nsteps), flat=True)

	return sampler, walkers, flat_chain

def run_mcmc_multiprocessing(config):
	burnin = config['mcmc']['burnin']
	nwalkers = config['mcmc']['nwalkers']
	nsteps = config['mcmc']['nsteps']
	ndim = config['ndim']
	save_dir = config['save_dir']

	initial_guess = np.array(config['initial_guess']) + 1e-3*np.random.randn(nwalkers, ndim)

	with Pool() as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, pool=pool)
		sampler.run_mcmc(initial_guess, nsteps, progress=True)
		walkers = sampler.get_chain(flat=False)
		np.save(f'{save_dir}/samples.npy', walkers)
		flat_chain = sampler.get_chain(discard=int(burnin*nsteps), flat=True)

	return sampler, walkers, flat_chain


def run_mcmc_MPI(config):
	burnin = config['mcmc']['burnin']
	nwalkers = config['mcmc']['nwalkers']
	nsteps = config['mcmc']['nsteps']
	niter = config['mcmc']['niter']
	ndim = config['ndim']
	save_dir = config['save_dir']

	walkers = []
	initial_guess = np.array(config['initial_guess']) + 1e-3*np.random.randn(nwalkers, ndim)

	with MPIPool() as pool:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)
		for i in range(niter):
			print('Iteration: ', i+1)
			print('\n')
			sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, pool=pool)
			sampler.run_mcmc(initial_guess, 300, progress=True)
			walkers.append(sampler.get_chain(flat=False))
			flat_chain = sampler.get_chain(flat=True)
			blobs = sampler.get_blobs(flat=True)
			idx = np.argmax(blobs)
			initial_guess = np.array(flat_chain[idx]) + 1e-3*np.random.randn(nwalkers, ndim)
			print('\n')

	# Now run long chain
		print('Final iteration')
		sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, pool=pool)
		sampler.run_mcmc(initial_guess, nsteps, progress=True)
		flat_chain = sampler.get_chain(flat=True, discard=int(nsteps*burnin))
		walkers.append(sampler.get_chain(flat=False))
		walkers = np.vstack(walkers)

		return sampler, walkers, flat_chain

def save_summary_plots(walkers, flat_chain, config):
	save_dir = config['save_dir']
	nwalkers = config['mcmc']['nwalkers']
	ndim = config['ndim']

	fig, ax = plt.subplots(ndim, 1, figsize=(5, 1.5*ndim))
	if ndim==1: ax = [ax]
	for i in range(nwalkers):
		for j in range(ndim):
			ax[j].plot(walkers[:, i, j])

	for i in range(ndim):
		ax[i].set_ylabel(config['latex_names'][i])

	plt.tight_layout()
	plt.savefig(f'{save_dir}/HMx_traceplot.pdf', dpi=300, bbox_inches='tight')

	fig = corner.corner(flat_chain, labels=config['latex_names'], show_titles=True)
	plt.savefig(f'{save_dir}/HMx_corner.pdf', dpi=300, bbox_inches='tight')


def save_best_fit(flat_chain, Pk_data, config, bfg_dict):
	bf_params = np.mean(flat_chain, axis=0)
	print('Best fit parameters: ', bf_params)
	h = bfg_dict['cosmo'].cosmo.params.h

	nfields = len(config['fields'])
	response_denom_theory = bfg_dict['P_mm_dmo'] if config['fit_response'] else 1
	response_denom_data = Pk_data['dmo']['Pk'] if config['fit_response'] else 1

	fig, ax = plt.subplots(2, nfields, figsize=(5*nfields, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
	if nfields == 1: ax = ax.reshape(2, 1)

	for i, field in enumerate(config['fields']):
		Pk_sim = Pk_data[field]['Pk']/response_denom_data
		k_sim = Pk_data[field]['k']
		variance = Pk_data[field]['variance']/response_denom_data**2

		# Pk_best fit
		Pk_theory, k = get_bfg_Pk(bf_params, config['fit_params'], field)
		# Pk_theory, k = get_bfg_Pk([], [], field)
		Pk_theory_interp = np.interp(k_sim, k/h, Pk_theory/response_denom_theory)

		# Pk_theory2, k = get_bfg_Pk([], [], field)
		# Pk_theory_interp2 = np.interp(k_sim, k/h, Pk_theory2/response_denom_theory)

		# import ipdb; ipdb.set_trace()
		ax[0, i].scatter(k_sim, Pk_sim,  alpha=0.7, s=4, c='k', label='Box2/hr')
		ax[0, i].plot(k_sim, Pk_theory_interp, ls='--', c='orangered', label='Best fit')
		# ax[0, i].plot(k_sim, Pk_theory_interp2, ls='--', c='limegreen', label='default')
		ax[0, i].set_xscale('log')
		ax[1, i].set_ylabel('R(k)')
		ax[1, i].set_xlabel('k [h/Mpc]')
		ax[0, i].text(0.1, 0.7, f'Pk {field}', c='gray',weight='semibold', transform=ax[0, i].transAxes)

		ax[1, i].plot(k_sim, Pk_theory_interp/Pk_sim - 1, ls='--', c='orangered')
		# ax[1, i].plot(k_sim, Pk_theory_interp2/Pk_sim - 1, ls='--', c='limegreen')
		ax[1, i].axhline(0, ls='--', c='gray')
		ax[1, i].set_ylim(-0.05, 0.05)
		ax[1, i].set_xscale('log')
		ax[1, i].set_xlabel('k [h/Mpc]')
		ax[1, i].set_ylabel('Ratio [Theory/Sim]')

	ax[0, 0].legend()
	plt.savefig(f'{config["save_dir"]}/best_fit_Pk.pdf', dpi=300, bbox_inches='tight')