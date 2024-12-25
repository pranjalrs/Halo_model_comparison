import corner
import emcee
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from schwimmbad import MPIPool
import sys

import pyccl as ccl
import pyhmcode
import pyhmcode.halo_profile_utils

from dawn.theory.power_spectrum import build_CAMB_cosmology

def get_hmcode_pk(param_values=None, fit_params=None, field=None):
	if field == 'matter-matter':
		hmcode_model = pyhmcode.Halomodel(pyhmcode.HMx2020_matter_w_temp_scaling)
		fields = [pyhmcode.field_matter]

	elif field == 'matter-pressure':
		hmcode_model = pyhmcode.Halomodel(pyhmcode.HMx2020_matter_pressure_w_temp_scaling)
		fields = [pyhmcode.field_matter, pyhmcode.field_electron_pressure]

	if param_values is not None:
		for i,name in enumerate(fit_params):
			if 'm0' in name:
				hmcode_model.__setattr__(name, 10**param_values[i])
			else:
				hmcode_model.__setattr__(name, param_values[i])

	hmcode_pofk = pyhmcode.calculate_nonlinear_power_spectrum(cosmology=hmcode_cosmology,
								halomodel=hmcode_model,
								fields=fields)

	return hmcode_pofk

def log_likelihood(param_values, fit_params, k_data, Pk_data, variance, field=None):
	for i in range(len(param_values)):
		prior = priors[i]
		if not (prior[0] <= param_values[i] <= prior[1]):
			return -np.inf

	hmcode_pofk = get_hmcode_pk(param_values, fit_params, field)
	# The output of calculate_nonlinear_power_spectrum has
	# shape (n_field, n_field, n_z, n_k).
	if field == 'matter-matter':
		Pk_theory = hmcode_pofk[0, 0, 0]

	elif field == 'matter-pressure':
		Pk_theory = hmcode_pofk[0, 1, 0]

	Pk_theory = np.interp(k_data, k, Pk_theory)
	# Compute the chi^2
	chi2 = np.sum((Pk_theory - Pk_data)**2/variance)

	return -0.5*chi2


#----------------------------------- 1. Setup Cosmology and Halo Model -----------------------------------#
#ccl_cosmology = ccl.CosmologyVanillaLCDM()
camb = build_CAMB_cosmology()
Pk_nonlin = camb.get_matter_power_interpolator(nonlinear=True)
params = camb.Params
ccl_cosmology = ccl.Cosmology(Omega_c=params.omegac, Omega_b=params.omegab, Omega_g=0, Omega_k=params.omk,
					h=params.h, sigma8=camb.get_sigma8_0(), n_s=camb.Params.InitPower.ns, Neff=params.N_eff, m_nu=0.0,
					w0=-1, wa=0, T_CMB=params.TCMB, transfer_function='boltzmann_camb', extra_parameters={'kmax':200.})


k = np.logspace(-2.5, 1.5, 100)
a = np.array([1]) #np.linspace(1/(1+6), 1, 10)
z = 1/a - 1

pofk_lin = np.array([ccl.linear_matter_power(ccl_cosmology, k=k, a=a_)
					for a_ in a])

# CCL uses units of Mpc, while pyhmcode uses Mpc/h. Hence we need to convert
# the units here.
h = ccl_cosmology["h"]
k = k/h
pofk_lin = pofk_lin * h**3

# Create the pyhmcode cosmology object. Beside the cosmology parameters, it
# also holds the linear power spectrum.
hmcode_cosmology = pyhmcode.halo_profile_utils.ccl2hmcode_cosmo(
						ccl_cosmo=ccl_cosmology,
						pofk_lin_k_h=k,
						pofk_lin_z=z[::-1],
						pofk_lin=pofk_lin[::-1],
						log10_T_heat=7.8)

#----------------------------------- 2. Load Data -----------------------------------#
# Load Pk for matter-matter
Pk_mm, box_size = np.loadtxt('../magneticum-data/data/Pylians/Pk_matter/Box2/Pk_hr_bao_CIC_R1024.txt'), 352

k_mm = Pk_mm[:, 0]
Pk_mm = Pk_mm[:, 1]
kmax = 6 # h/Mpc
Pk_mm,	k_mm = Pk_mm[k_mm < kmax], k_mm[k_mm < kmax]

delta_k = 2*np.pi/box_size
Nk = 2*np.pi * (k_mm/delta_k)**2
variance_Pk_mm = Pk_mm**2/Nk

# Load Pk for matter-pressure
Pk_mp, box_size = np.loadtxt('../magneticum-data/data/Pylians/Pk_matterxpressure/Box2_CIC_R1024.txt'), 352

k_mp = Pk_mp[:, 0]
Pk_mp = Pk_mp[:, 1]
kmax = 6 # h/Mpc
Pk_mp,	k_mp = Pk_mp[k_mp < kmax], k_mp[k_mp < kmax]

delta_k = 2*np.pi/box_size
Nk = 2*np.pi * (k_mp/delta_k)**2
variance_Pk_mp = Pk_mm**2/Nk

fit_params_mm = ['eps_array', 'gamma_array', 'm0_array']
priors_mm = [[-0.95, 3], [1.05, 3], [10, 17]]
initial_parameters_mm = [0.2038, 1.33, 13.3]


fit_params_mp = ['eps_array', 'gamma_array', 'm0_array', 'alpha_array', 'twhim_array']
priors_mp = [[-0.95, 3], [1.05, 3], [10, 17], [0, 1], [6, 7]]
initial_parameters_mp = [0.2038, 1.33, 13.3, 0.84, 6.65]

if __name__=='__main__':

	# Use first input argument from sys.argv to determine which field to fit
	field = sys.argv[1] # 'matter-matter' or 'matter-pressure'
	parallel = sys.argv[2]  # 'multiprocessing' or 'MPI'

	if field == 'matter-matter':
		fit_params = fit_params_mm
		initial_parameters = initial_parameters_mm
		priors = priors_mm
		k_sim, Pk_sim, variance = k_mm, Pk_mm, variance_Pk_mm
		Pk_label = '$P_{mm}(k)$ [Mpc/$h]^3$'
		save_dir = 'Pk_mm_fit'


	elif field == 'matter-pressure':
		fit_params = fit_params_mp
		initial_parameters = initial_parameters_mp
		priors = priors_mp
		k_sim, Pk_sim, variance = k_mp, Pk_mp, variance_Pk_mp
		Pk_label = '$P_{mp}(k)$ [ev/cm$^3$] (Mpc/h)$^3$'
		save_dir = 'Pk_mp_fit'

	ndim = len(initial_parameters)
	nwalkers = 5*len(fit_params)
	nsteps = 5000
	# Initialize the walkers
	initial = np.array(initial_parameters) + 1e-3*np.random.randn(nwalkers, ndim)

	#----------------------------------- 4. Run emcee -----------------------------------#

	if parallel == 'multiprocessing':
		with Pool() as pool:
			sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(fit_params, k_sim, Pk_sim, variance, field), pool=pool)
			sampler.run_mcmc(initial, nsteps, progress=True)
			walkers = sampler.get_chain(flat=False)
			np.save('chains/Pk_mm_fit/HMx.npy', walkers)
			flat_chain = sampler.get_chain(discard=int(0.8*nsteps), flat=True)

	elif parallel == 'MPI':
		walkers = []
		with MPIPool() as pool:
			if not pool.is_master():
				pool.wait()
				sys.exit(0)
			for i in range(1):
				print('Iteration: ', i+1)
				print('\n')
				sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(fit_params, k_sim, Pk_sim, variance, field), pool=pool)
				sampler.run_mcmc(initial, 500, progress=True)
				walkers.append(sampler.get_chain(flat=False))
				flat_chain = sampler.get_chain(flat=True)
				blobs = sampler.get_blobs(flat=True)
				idx = np.argmax(blobs)
				initial = np.array(flat_chain[idx]) + 1e-3*np.random.randn(nwalkers, ndim)

	# Now run long chain
			print('Final iteration')
			sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(k_sim, Pk_sim), pool=pool)
			sampler.run_mcmc(initial, nsteps, progress=True)
			flat_chain = sampler.get_chain(flat=True, discard=int(nsteps*0.7))
			walkers.append(sampler.get_chain(flat=False))
			walkers = np.vstack(walkers)
			np.save(f'chains/{save_dir}/HMx.npy', walkers)

	fig = corner.corner(flat_chain, labels=fit_params, show_titles=True)
	plt.savefig(f'chains/{save_dir}/HMx_corner.pdf', dpi=300, bbox_inches='tight')

	#----------------------------------- 5. Plot best fit-----------------------------------#
	best_fit = np.mean(flat_chain, axis=0)

	best_fit_Pk = get_hmcode_pk(best_fit)
	best_fit_Pk = np.interp(k_sim, k, best_fit_Pk[0,0,0])

	fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharex=False, gridspec_kw={'height_ratios': [3, 1]})

	ax[0, 0].loglog(k_sim, Pk_sim, c='dodgerblue', label='Simulation')
	# Draw shaded region to indicate error
	ax[0, 0].fill_between(k_sim, Pk_sim - np.sqrt(variance), Pk_sim + np.sqrt(variance), color='lightgray', alpha=0.5)
	ax[0, 0].loglog(k_sim, best_fit_Pk, c='red', ls='--', label='HMx: Best fit')
	ax[0, 0].set_ylabel(Pk_label)
	ax[0, 0].legend()

	ax[1, 0].semilogx(k_sim, best_fit_Pk/Pk_sim, c='red', ls='--', label='Best fit/Simulation')


	ax[1, 0].fill_between(k_sim, 1 - np.sqrt(variance)/Pk_sim, 1 + np.sqrt(variance)/Pk_sim, color='lightgray', alpha=0.5)
	ax[1, 0].axhline(1, c='gray', ls='--')
	ax[1, 0].set_ylim(0.8, 1.1)
	ax[1, 0].set_xlabel('k [h/Mpc]')
	ax[1, 0].set_ylabel('Ratio [Theory/Simulation]')

	plt.savefig(f'chains/{save_dir}/HMx_bestfit.pdf', dpi=300, bbox_inches='tight')

