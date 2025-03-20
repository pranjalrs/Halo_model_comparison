import corner
import emcee
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from schwimmbad import MPIPool
import sys
import ultranest.stepsampler

import astropy.units as u
import astropy.constants as const
import pyccl as ccl
import BaryonForge as bfg

import utils

Pk_data = {}
config = {}
bfg_dict = {}

Mmax = 2e15/0.704
Mmin = 1e9

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



def init_bfg_model(halo_model, cosmo, a, k=None):
	k = np.logspace(-2, 1., 100) if k is None else k  # in 1/Mpc
	a = a
	cosmo = cosmo

	rho  = ccl.rho_x(cosmo, 1, 'matter', is_comoving = True)
	fft_precision = dict(padding_lo_fftlog = 1e-8, padding_hi_fftlog = 1e8, n_per_decade = 100)
	mdef = ccl.halos.MassDef(Delta="200", rho_type="critical")
	c_M_relation = ccl.halos.concentration.ConcentrationDiemer15


	if halo_model == 'Arico20':
		DMO = bfg.Profiles.Arico20.DarkMatter(c_M_relation=c_M_relation) / rho

	elif halo_model == 'Mead20':
		DMO = bfg.Profiles.Mead20.DarkMatter(c_M_relation=c_M_relation, mass_def = mdef) / rho


	elif halo_model == 'Schneider19':
		## TO DO: Fix computation for Schneider19
		DMO = bfg.Profiles.Schneider19.DarkMatter(c_M_relation=c_M_relation, epsilon = 4, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)
		M_2_Mtot = bfg.Profiles.misc.Mdelta_to_Mtot(DMO, r_min = 1e-6, r_max = 1e2, N_int = 100)
		## Only convert to contrast after computing M_2_Mtot!!
		DMO = DMO/rho



	#We will use the built-in, CCL halo model calculation tools.
	hmf = ccl.halos.MassFuncTinker10(mass_def=mdef, mass_def_strict = False)
	hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef, mass_def_strict = False)
	HMC  = ccl.halos.halo_model.HMCalculator(mass_function = hmf, halo_bias = hbf,
										mass_def = mdef,
										log10M_min = np.log10(Mmin), log10M_max = np.log10(Mmax), nM = 100)

	HMC_mod  = utils.HMCalculator_NoCorrection(mass_function = hmf, halo_bias = hbf,
                                    mass_def = mdef,
                                    log10M_min = np.log10(1e12), log10M_max = np.log10(Mmax), nM = 100)

	HOD_profile =  ccl.halos.profiles.hod.HaloProfileHOD(mass_def=mdef, concentration=c_M_relation(mass_def=mdef))


	if halo_model == 'Schneider19':
		print('[INFO]: Using Flexible HM Calculator for Schneider19')
		HMC = bfg.utils.FlexibleHMCalculator(mass_function = hmf, halo_bias = hbf, halo_m_to_mtot = M_2_Mtot,
										  mass_def = mdef,
										  log10M_min = np.log10(Mmin), log10M_max = np.log10(Mmax), nM = 100)

		HMC_mod = utils.HMCalculator_NoCorrection_Flexible(mass_function = hmf, halo_bias = hbf, halo_m_to_mtot = M_2_Mtot,
										  mass_def = mdef,
										  log10M_min = np.log10(Mmin), log10M_max = np.log10(Mmax), nM = 100)


	# Compute DMO power spectrum for response calculation
	DMO.update_precision_fftlog(**fft_precision)
	P_mm_dmo = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k, a, DMO)*cosmo.cosmo.params.h**3
	Pk_lin = ccl.pk2d.parse_pk(cosmo)(k, a)
	# Pressure units of data is eV/cm^3
	cgs_to_eV_cm3__factor = (u.erg/u.cm**3).to(u.eV/u.cm**3)


	bfg_meta_dict = {'halo_model': halo_model,
					 'k': k,
					 'a': a,
					 'cosmo': cosmo,
					 'rho': rho,
					 'fft_precision': fft_precision,
					 'mdef': mdef,
					 'hbf': hbf,
					 'c_M_relation': c_M_relation,
					 'HMC': HMC,
					 'HMC_mod': HMC_mod,
					 'HOD_profile': HOD_profile,
					 'P_mm_dmo': P_mm_dmo,
					 'Pk_lin': Pk_lin,
					 'cgs_to_eV_cm3__factor': cgs_to_eV_cm3__factor}

	return bfg_meta_dict

def transition_alpha(alpha=None):
	if alpha is not None:
		return lambda a: alpha

	else:
		return None


def get_bfg_Pk(par, field=None, param_dict=None):
	cosmo = bfg_dict['cosmo']
	h = cosmo.cosmo.params.h
	f_baryon = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
	smooth_transition_p = transition_alpha(par.get('transition_alpha', None))
	smooth_transition_ne = transition_alpha(par.get('transition_alpha_ne', None))

	c_M_relation = bfg_dict['c_M_relation']
	mdef = bfg_dict['mdef']
	HMC = bfg_dict['HMC']
	HMC_mod = bfg_dict['HMC_mod']
	cosmo = bfg_dict['cosmo']
	k = bfg_dict['k']
	a = bfg_dict['a']

	if bfg_dict['halo_model'] == 'Mead20':
		#Define profiles. Normalize to convert density --> overdensity
		DMB = bfg.Profiles.Mead20.DarkMatterBaryon(**par, c_M_relation=c_M_relation, mass_def = mdef)/bfg_dict['rho']
		PRS = bfg.Profiles.Mead20.Pressure(**par, c_M_relation=c_M_relation, mass_def=mdef)
		GAS = bfg.Profiles.Mead20.Gas(**par, c_M_relation=c_M_relation, mass_def=mdef)
		ElectronDensity = bfg.Profiles.GasNumberDensity(gas = GAS, mean_molecular_weight = 1.14, mass_def=mdef) #simple constant rescaling of gas density --> number density in cgs
		ElectronDensity_sq = ElectronDensity**2

		delta_ElectronDensity = ElectronDensity / bfg_dict['rho']

		for p in [DMB, PRS, GAS, ElectronDensity, ElectronDensity_sq, delta_ElectronDensity]:
			p.update_precision_fftlog(**bfg_dict['fft_precision'])

	elif bfg_dict['halo_model'] == 'Arico20':
		DMO = bfg.Profiles.Arico20.DarkMatter()
		T   = bfg.Profiles.misc.Truncation(epsilon = 1)
		DMO = bfg.Profiles.Arico20.DarkMatter(**par)
		GAS = bfg.Profiles.Arico20.Gas(**par, r_min_int = 1e-5, r_max_int = 1e2, r_steps = 500)
		STR = bfg.Profiles.Arico20.Stars(**par, r_min_int = 1e-6, r_max_int = 10, r_steps = 500)
		CLM = bfg.Profiles.Arico20.CollisionlessMatter(**par, darkmatter = DMO, max_iter = 5, reltol = 5e-2, r_steps = 100)
		DMB = bfg.Profiles.Arico20.DarkMatterBaryon(gas = GAS, stars = STR, collisionlessmatter = CLM)
		ElectronDensity = bfg.Profiles.GasNumberDensity(gas = GAS, mean_molecular_weight = 1.14, mass_def=bfg_dict['mdef']) #simple constant rescaling of gas density --> number density in cgs
		ElectronDensity_sq = ElectronDensity**2


		PRS = bfg.Profiles.Arico20.ThermalPressure(**par)
		#Now convert density --> overdensity
		DMB = DMB / bfg_dict['rho']
		DMO = DMO / bfg_dict['rho']

		delta_ElectronDensity = ElectronDensity / bfg_dict['rho']
		#Upgrade precision of all profiles.
		for p in [DMB, DMO, PRS, ElectronDensity, ElectronDensity_sq, delta_ElectronDensity]:
			p.update_precision_fftlog(**bfg_dict['fft_precision'])

	elif bfg_dict['halo_model'] == 'Schneider19':
		#For Schneider19, the DarkMatterBaryon profile includes the TwoHalo term.
		#So let's make sure to remove that. The ``DarkMatterBaryon'' class is
		#still good one to use since it makes sure Gas + star + DM is normalized
		#to match the total mass from the DMO case
		DMO = bfg.Profiles.Schneider19.DarkMatter(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)
		GAS = bfg.Profiles.Schneider19.Gas(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)
		STR = bfg.Profiles.Schneider19.Stars(**par, r_min_int = 1e-6, r_max_int = 5, r_steps = 500)
		CLM = bfg.Profiles.Schneider19.CollisionlessMatter(**par, max_iter = 5, reltol = 5e-2, r_steps = 500)
		ElectronDensity = bfg.Profiles.GasNumberDensity(gas = GAS, mean_molecular_weight = 1.14, mass_def=bfg_dict['mdef']) #simple constant rescaling of gas density --> number density in cgs
		ElectronDensity_sq = ElectronDensity**2

		DMB = bfg.Profiles.Schneider19.DarkMatterBaryon(**par,
												gas = GAS, stars = STR,
												collisionlessmatter = CLM, darkmatter = DMO,
												twohalo = bfg.Profiles.misc.Zeros(),
												r_steps = 500)

		PRS = bfg.Profiles.Pressure(gas = GAS, darkmatterbaryon = DMB, **par, r_min_int = 1e-4, r_max_int = 1e2, r_steps = 500)
		NonThermalPRS = bfg.Profiles.NonThermalFrac(**par)
		PRS = PRS * (1 - NonThermalPRS)

		#Now convert density --> overdensity
		#We do this later because PRS is defined with density
		DMB = DMB / bfg_dict['rho']
		DMO = DMO / bfg_dict['rho']

		T   = bfg.Profiles.misc.Truncation(epsilon = 100)
		DMB = DMB * T
		DMO = DMO * T


		delta_ElectronDensity = ElectronDensity / bfg_dict['rho']

		# Upgrade precision of all profiles.
		for p in [DMB, DMO, PRS, ElectronDensity, ElectronDensity_sq, delta_ElectronDensity]:
			p.update_precision_fftlog(**bfg_dict['fft_precision'])


	if field == 'm-m':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k, a, DMB)*h**3

	elif field == 'm-p':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k, a,
											   DMB, prof2 = PRS, smooth_transition=smooth_transition_p)*h**3
		Pk = Pk*bfg_dict['cgs_to_eV_cm3__factor']

	elif field == 'ne-ne':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k, a,
											   ElectronDensity, smooth_transition=smooth_transition_ne)*h**3

	elif field == 'g-ne':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k, a,
											   ElectronDensity, prof2=bfg_dict['HOD_profile'], smooth_transition=smooth_transition_ne)*h**3

	elif field == 'h-ne':
		# Use HMC_mod to do integral over a subset of halos
		Pk_1h = HMC_mod.I_0_1(cosmo, k, a, ElectronDensity)

		bias = lambda mass: bfg_dict['hbf'](cosmo=bfg_dict['cosmo'], M=mass, a=bfg_dict['a'])
		integral_bias = HMC_mod.integrate_over_massfunc(bias, cosmo, a)  # Integrate halo bias over subset of halos
		integral_prof = HMC.I_1_1(cosmo, k, a, ElectronDensity)

		Pk_2h = bfg_dict['Pk_lin'] * integral_bias * integral_prof

		alpha = 1 if smooth_transition_ne is None else smooth_transition_ne(a)
		norm = ElectronDensity.get_normalization(cosmo, a, hmc=HMC)
		Pk = (Pk_1h**alpha + Pk_2h**alpha)**(1/alpha) / norm
		Pk = Pk * h**3

	elif field == 'frb':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k, a,
											   delta_ElectronDensity, smooth_transition=smooth_transition)*h**3

	elif field == 'xray':
		Pk = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k, a,
											   ElectronDensity_sq, smooth_transition=smooth_transition)*h**3

    # Return Pk in [field units]x[Mpc/h]**3 and k in k/h
	return Pk, k/h


def log_likelihood(param_values, return_total=True, apply_prior=True):
	if apply_prior:
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
	log_like_field = []
    
	par = utils.get_param_dict(param_values, config['fit_params'], param_dict=None, halo_model = bfg_dict['halo_model'])

	for field in config['fields']:
		try:
			Pk_theory, k = get_bfg_Pk(par, field)
		except ValueError:
			return -1e100

		Pk_sim = Pk_data[field]['Pk']/response_denom_data
		k_sim = Pk_data[field]['k']
		variance = Pk_data[field]['variance']/response_denom_data**2

		# Note that CCL uses Mpc units for k; need to convert to h/Mpc
		Pk_theory_interp = np.interp(k_sim, k/bfg_dict['cosmo'].cosmo.params.h, Pk_theory/response_denom_theory)

		this_log_likelihood = -0.5*np.sum((Pk_theory_interp - Pk_sim)**2/variance)
		log_like_field.append(this_log_likelihood)
		log_likelihood += this_log_likelihood

	if return_total:
		if np.isnan(log_likelihood):
			return -1e100
		return log_likelihood

	else:
		return log_like_field


def prior_transform(cube, config):
	params = np.zeros_like(cube)
	for i in range(config['ndim']):
		prior = config['priors'][i]
		low_lim, up_lim = prior[0], prior[1]
		params[i] = cube[i] * (up_lim - low_lim) + low_lim

	return params


def run_ultranest(config, run_num=1):
	log_likelihood_ultranest = lambda x: log_likelihood(x, return_total=True, apply_prior=False)
	prior_transform_ultranest = lambda x: prior_transform(x, config)

	sampler = ultranest.ReactiveNestedSampler(config['fit_params'], log_likelihood_ultranest, prior_transform_ultranest,
					    log_dir=config['save_dir'], run_num=run_num, resume=True)
	sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=2*len(config['fit_params']),
					       generate_direction=ultranest.stepsampler.generate_mixture_random_direction)

	sampler.run(min_num_live_points=400, show_status=True)
	# sampler.run(min_num_live_points=4, max_iters=1, show_status=True)

	return sampler


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
	plt.savefig(f'{save_dir}/traceplot.pdf', dpi=300, bbox_inches='tight')

	fig = corner.corner(flat_chain, labels=config['latex_names'], show_titles=True)
	plt.savefig(f'{save_dir}/corner.pdf', dpi=300, bbox_inches='tight')


def save_best_fit(bf_params, Pk_data, config, bfg_dict):
	print('Best fit parameters: ', bf_params)
	h = bfg_dict['cosmo'].cosmo.params.h

	nfields = len(config['fields'])
	response_denom_theory = bfg_dict['P_mm_dmo'] if config['fit_response'] else 1
	response_denom_data = Pk_data['dmo']['Pk'] if config['fit_response'] else 1

	# log likelihood
	log_like = log_likelihood(bf_params, return_total=False)

	par_dict = utils.get_param_dict(bf_params, config['fit_params'], param_dict=None, halo_model = bfg_dict['halo_model'])

	fig, ax = plt.subplots(2, nfields, figsize=(5*nfields, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
	if nfields == 1: ax = ax.reshape(2, 1)

	for i, field in enumerate(config['fields']):
		Pk_sim = Pk_data[field]['Pk']/response_denom_data
		k_sim = Pk_data[field]['k']
		variance = Pk_data[field]['variance']/response_denom_data**2

		# Pk_best fit
		Pk_theory, k = get_bfg_Pk(par_dict, field)
		# Pk_theory, k = get_bfg_Pk([], [], field)
		Pk_theory_interp = np.interp(k_sim, k/h, Pk_theory/response_denom_theory)

		# Pk_theory2, k = get_bfg_Pk([], [], field)
		# Pk_theory_interp2 = np.interp(k_sim, k/h, Pk_theory2/response_denom_theory)

		text = f'Pk {field}\n $\chi^2$: {-2*log_like[i]:.2f}\n $\chi^2_\\nu$ {-2*log_like[i]/(len(k_sim)-len(bf_params)):.2f}'

		if field == 'm-p':
			factor = 1e3
			ylabel = ' $\\times 10^3$'
			text_y = 0.7
		else:
			factor = 1
			ylabel = ''
			text_y = 0.2

		ax[0, i].scatter(k_sim, Pk_sim*factor,  alpha=0.7, s=4, c='k', label='Box2b/hr')
		ax[0, i].plot(k_sim, Pk_theory_interp*factor, ls='--', c='orangered', label='Best fit')
		ax[0, i].fill_between(k_sim, Pk_sim*factor - np.sqrt(variance)*factor, Pk_sim*factor + np.sqrt(variance)*factor,
						color='gray', alpha=0.3, linewidth=0)
		# ax[0, i].plot(k_sim, Pk_theory_interp2, ls='--', c='limegreen', label='default')
		ax[0, i].set_xscale('log')
		ax[0, i].set_ylabel('$R_{\mathrm{'+field+'}}(k)$'+ylabel)
		ax[1, i].set_xlabel('k [h/Mpc]')
		ax[0, i].text(0.1, text_y, text, c='gray', transform=ax[0, i].transAxes)

		ax[1, i].plot(k_sim, Pk_theory_interp/Pk_sim - 1, ls='--', c='orangered')
		ax[1, i].fill_between(k_sim, -np.sqrt(variance)/Pk_sim, np.sqrt(variance)/Pk_sim, color='gray', alpha=0.3, linewidth=0)
		# ax[1, i].plot(k_sim, Pk_theory_interp2/Pk_sim - 1, ls='--', c='limegreen')
		ax[1, i].axhline(0, ls='--', c='gray')
		ax[1, i].set_ylim(-0.05, 0.05)
		ax[1, i].set_xscale('log')
		ax[1, i].set_xlabel('k [h/Mpc]')
		ax[1, i].set_ylabel('$\Delta R(k)/R_\mathrm{sim}(k)$')

	ax[0, 0].legend(loc='upper right')
	return fig
