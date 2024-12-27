import argparse
import corner
import emcee
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os

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

	hmcode_model.mmax = 2e15
	if param_values is not None:
		for i,name in enumerate(fit_params):
			if 'm0' in name or 'twhim' in name:
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
	if field == 'matter-matter':
		Pk_theory = hmcode_pofk[0, 0, 0]

	elif field == 'matter-pressure':
		Pk_theory = hmcode_pofk[0, 1, 0]

	Pk_theory = np.interp(k_data, k, Pk_theory)
	# Compute the chi^2
	chi2 = np.sum((Pk_theory - Pk_data)**2/variance)

	return -0.5*chi2

def joint_log_likelihood(param_values, fit_params, k_data_mm, Pk_data_mm, variance_mm, k_data_mp, Pk_data_mp, variance_mp, fit_response=False):
	for i in range(len(param_values)):
		prior = priors[i]
		if not (prior[0] <= param_values[i] <= prior[1]):
			return -np.inf

	hmcode_pofk = get_hmcode_pk(param_values, fit_params, 'matter-pressure')

	response_factor_hmx = hmcode_pofk_dmo if fit_response else 1
	response_factor_data = Pk_mm_dmo if fit_response else 1

	Pk_theory_mm = np.interp(k_data_mm, k, hmcode_pofk[0, 0, 0]/response_factor_hmx)
	Pk_theory_mp = np.interp(k_data_mp, k, hmcode_pofk[0, 1, 0]/response_factor_hmx)

	Pk_data_mm = Pk_data_mm/response_factor_data
	Pk_data_mp = Pk_data_mp/response_factor_data
	variance_mm = variance_mm/response_factor_data**2
	variance_mp = variance_mp/response_factor_data**2

	# Compute the chi^2
	chi2 = np.sum((Pk_theory_mm - Pk_data_mm)**2/variance_mm) + np.sum((Pk_theory_mp - Pk_data_mp)**2/variance_mp)

	return -0.5*chi2

def plot_Pk(ax1, ax2, k_sim, Pk_sim, variance, Pk_bf, Pk_default, Pk_label):
	ax1.scatter(k_sim, Pk_sim, s=4, c='k', label='Simulation')
	# Draw shaded region to indicate error
	ax1.fill_between(k_sim, Pk_sim - np.sqrt(variance), Pk_sim + np.sqrt(variance), color='lightgray', alpha=0.5)
	ax1.plot(k_sim, Pk_bf, c='red', ls='--', label='HMx: Best fit')
	ax1.plot(k_sim, Pk_default, c='limegreen', label='HMx: Default')
	ax1.set_ylabel(Pk_label)
	ax1.set_xscale('log')
	ax1.legend()

	ax2.semilogx(k_sim, Pk_default/Pk_sim, c='limegreen')
	ax2.semilogx(k_sim, Pk_bf/Pk_sim, c='red', ls='--')


	ax2.fill_between(k_sim, 1 - np.sqrt(variance)/Pk_sim, 1 + np.sqrt(variance)/Pk_sim, color='lightgray', alpha=0.5)
	ax2.axhline(1, c='gray', ls='--')
	ax2.set_ylim(0.8, 1.2)
	ax2.set_xlabel('k [h/Mpc]')
	ax2.set_ylabel('Ratio [HMx/Sim]')

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

hmcode_cosmology_dmo = pyhmcode.halo_profile_utils.ccl2hmcode_cosmo(
						ccl_cosmo=ccl_cosmology,
						pofk_lin_k_h=k,
						pofk_lin_z=z[::-1],
						pofk_lin=pofk_lin[::-1],
						log10_T_heat=0.)

hmcode_model = pyhmcode.Halomodel(pyhmcode.HMx2020_matter_pressure_w_temp_scaling)
hmcode_pofk_dmo = pyhmcode.calculate_nonlinear_power_spectrum(cosmology=hmcode_cosmology_dmo,
                            halomodel=hmcode_model, fields=[pyhmcode.field_matter])[0, 0, 0]
#----------------------------------- 2. Load Data -----------------------------------#
# Load Pk for matter-matter
Pk_mm, box_size = np.loadtxt('../magneticum-data/data/Pylians/Pk_matter/Box2/Pk_hr_bao_CIC_R1024.txt'), 352
Pk_mm_dmo = np.loadtxt('../magneticum-data/data/Pylians/Pk_matter/Box2/Pk_hr_dm_CIC_R1024.txt')
Pk_mm_dmo = Pk_mm_dmo[:, 1]

k_mm = Pk_mm[:, 0]
Pk_mm = Pk_mm[:, 1]
kmax = 6 # h/Mpc
Pk_mm_dmo = Pk_mm_dmo[k_mm < kmax]
Pk_mm, k_mm = Pk_mm[k_mm < kmax], k_mm[k_mm < kmax]

delta_k = 2*np.pi/box_size
Nk = 2*np.pi * (k_mm/delta_k)**2
variance_Pk_mm = Pk_mm**2/Nk

# Load Pk for matter-pressure
Pk_mp, box_size = np.loadtxt('../magneticum-data/data/Pylians/Pk_matterxpressure/Box2_CIC_R1024.txt'), 352

k_mp = Pk_mp[:, 0]
Pk_mp = Pk_mp[:, 1]*1e3  # Convert to eV/cm^3
kmax = 6 # h/Mpc
Pk_mp,	k_mp = Pk_mp[k_mp < kmax], k_mp[k_mp < kmax]

delta_k = 2*np.pi/box_size
Nk = 2*np.pi * (k_mp/delta_k)**2
variance_Pk_mp = Pk_mp**2/Nk

fit_params_mm = ['eps_array', 'gamma_array', 'm0_array']
priors_mm = [[-0.95, 3], [1.05, 3], [10, 17]]
initial_parameters_mm = [0.2038, 1.33, 13.3]


fit_params_mp = ['eps_array', 'gamma_array', 'm0_array', 'alpha_array', 'twhim_array', 'eps2_array']
latex_names_mp = ['$\epsilon_1$', '$\Gamma$', '$\log M_0$', '$\\alpha$', '$T_{WHIM}$', '$\epsilon_2$']
priors_mp = [[-0.95, 3], [1.05, 3], [10, 17], [0, 1.5], [6, 7.5], [-0.95, 3]]
initial_parameters_mp = [0.2038, 1.33, 13.3, 0.84, 6.65, 0.2]

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--field', choices=['matter-matter', 'matter-pressure', 'joint'], type=str)
	parser.add_argument('--mcmc_args', default='multiprocessing', type=str)
	parser.add_argument('--nparam', default=None, type=int)
	parser.add_argument('--fit_response', default=0, type=int)
	args = parser.parse_args()
	# Use first input argument from sys.argv to determine which field to fit
	field = args.field # 'matter-matter' or 'matter-pressure'
	parallel = args.mcmc_args  # 'multiprocessing' or 'MPI'
	nparam = args.nparam
	fit_response = bool(args.fit_response)

	if field == 'matter-matter':
		# If no. of fit params is not specified use all        
		if nparam is None: nparam = len(fit_params_mm)
		fit_params = fit_params_mm[:nparam]
		initial_parameters = initial_parameters_mm[:nparam]
		priors = priors_mm[:nparam]
		k_sim, Pk_sim, variance = k_mm, Pk_mm, variance_Pk_mm
		Pk_label = '$P_{mm}(k)$ [Mpc/$h]^3$'
		save_dir = f'chains/Pk_mm_fit_nparam_{nparam}'
		likelihood = log_likelihood
		likelihood_args =  (fit_params, k_sim, Pk_sim, variance, field)

	elif field == 'matter-pressure':
		if nparam is None: nparam = len(fit_params_mp)
		fit_params = fit_params_mp[:nparam]
		latex_names = latex_names_mp[:nparam]
		initial_parameters = initial_parameters_mp[:nparam]
		priors = priors_mp[:nparam]
		k_sim, Pk_sim, variance = k_mp, Pk_mp, variance_Pk_mp
		Pk_label = '$P_{mp}(k)$ [ev/cm$^3$] (Mpc/h)$^3$'
		save_dir = f'chains/Pk_mp_fit_nparam_{nparam}'
		likelihood = log_likelihood
		likelihood_args =  (fit_params, k_sim, Pk_sim, variance, field)

	elif field == 'joint':
		if nparam is None: nparam = len(fit_params_mp)
		fit_params = fit_params_mp[:nparam]
		latex_names = latex_names_mp[:nparam]
		initial_parameters = initial_parameters_mp[:nparam]
		priors = priors_mp[:nparam]
		Pk_label_mm = '$P_{mm}(k)$ [Mpc/$h]^3$'
		Pk_label_mp = '$P_{mp}(k)$ [ev/cm$^3$] (Mpc/h)$^3$'
		save_dir = f'chains/Pk_joint_fit_nparam_{nparam}'
		if fit_response: save_dir = f'chains/Pk_joint_fit_nparam_{nparam}_fit_response'
		likelihood = joint_log_likelihood
		likelihood_args =  (fit_params, k_mm, Pk_mm, variance_Pk_mm, k_mp, Pk_mp, variance_Pk_mp, fit_response)

	else:
		print(f'Field {field} is not a valid option!')
		sys.exit()

	if not os.path.exists(save_dir):        
		try:
		#Needed when using MPI and creating dir
			os.makedirs(save_dir)
			print(f"Directory created: {save_dir}")
		except FileExistsError:
			pass

	ndim = len(initial_parameters)
	nwalkers = 5*len(fit_params)
	nsteps = 2500
	# Initialize the walkers
	initial = np.array(initial_parameters) + 1e-3*np.random.randn(nwalkers, ndim)

	#----------------------------------- 4. Run emcee -----------------------------------#

	if parallel == 'multiprocessing':
		with Pool() as pool:
			sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=likelihood_args, pool=pool)
			sampler.run_mcmc(initial, nsteps, progress=True)
			walkers = sampler.get_chain(flat=False)
			np.save(f'{save_dir}/HMx.npy', walkers)
			flat_chain = sampler.get_chain(discard=int(0.8*nsteps), flat=True)

	elif parallel == 'MPI':
		walkers = []
		with MPIPool() as pool:
			if not pool.is_master():
				pool.wait()
				sys.exit(0)
			for i in range(3):
				print('Iteration: ', i+1)
				print('\n')
				sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=likelihood_args, pool=pool)
				sampler.run_mcmc(initial, 500, progress=True)
				walkers.append(sampler.get_chain(flat=False))
				flat_chain = sampler.get_chain(flat=True)
				blobs = sampler.get_blobs(flat=True)
				idx = np.argmax(blobs)
				initial = np.array(flat_chain[idx]) + 1e-3*np.random.randn(nwalkers, ndim)
				print('\n')

	# Now run long chain
			print('Final iteration')
			sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=likelihood_args, pool=pool)
			sampler.run_mcmc(initial, nsteps, progress=True)
			flat_chain = sampler.get_chain(flat=True, discard=int(nsteps*0.8))
			walkers.append(sampler.get_chain(flat=False))
			walkers = np.vstack(walkers)
			np.save(f'{save_dir}/HMx.npy', walkers)

	if parallel == 'load_samples':
		walkers = np.load(f'{save_dir}/HMx.npy')
		nsteps, nwalkers, ndim = walkers.shape
		# Discard 80% and flatten chain
		flat_chain = walkers[int(0.8*nsteps):, :, :].reshape((int(0.2*nsteps)*nwalkers, ndim))

	# Save Trace plot
	fig, ax = plt.subplots(nparam, 1, figsize=(5, 1.5*nparam))
	if nparam==1: ax = [ax]    
	for i in range(nwalkers):
		for j in range(len(fit_params)):
			ax[j].plot(walkers[:, i, j])

	for i in range(nparam):
		ax[i].set_ylabel(latex_names[i])

	plt.tight_layout()
	plt.savefig(f'{save_dir}/HMx_traceplot.pdf', dpi=300, bbox_inches='tight')

	fig = corner.corner(flat_chain, labels=fit_params, show_titles=True)
	plt.savefig(f'{save_dir}/HMx_corner.pdf', dpi=300, bbox_inches='tight')

	#----------------------------------- 5. Plot best fit-----------------------------------#
	best_fit = np.mean(flat_chain, axis=0)

    
	if field != 'joint':
		Pk_bf = get_hmcode_pk(best_fit, fit_params, field)
		Pk_default = get_hmcode_pk(field=field)
		fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharex=False, gridspec_kw={'height_ratios': [3, 1]})
		if field == 'matter-matter':
			Pk_bf = np.interp(k_sim, k, Pk_bf[0, 0, 0])
			Pk_default = np.interp(k_sim, k, Pk_default[0, 0, 0])


		elif field == 'matter-pressure':
			Pk_bf = np.interp(k_sim, k, Pk_bf[0, 1, 0])
			Pk_default = np.interp(k_sim, k, Pk_default[0, 1, 0])

		plot_Pk(ax1, ax2, k_sim, Pk_sim, variance, Pk_bf, Pk_default, Pk_label)

	else:
		Pk_bf = get_hmcode_pk(best_fit, fit_params, 'matter-pressure')
		Pk_default = get_hmcode_pk(field='matter-pressure')
		Pk_HMx_dmo = np.interp(k_mm, k, hmcode_pofk_dmo)

		fig, ax = plt.subplots(4, 2, figsize=(15, 12), sharex=False, gridspec_kw={'height_ratios': [3, 1, 3, 1]})
		Pk_bf_mm = np.interp(k_mm, k, Pk_bf[0, 0, 0])
		Pk_default_mm = np.interp(k_mm, k, Pk_default[0, 0, 0])
		plot_Pk(ax[0, 0], ax[1, 0], k_mm, Pk_mm, variance_Pk_mm, Pk_bf_mm, Pk_default_mm, Pk_label_mm)
		plot_Pk(ax[0, 1], ax[1, 1], k_mm, Pk_mm/Pk_mm_dmo, variance_Pk_mm/Pk_mm_dmo**2, 
                Pk_bf_mm/Pk_HMx_dmo, Pk_default_mm/Pk_HMx_dmo, '$R_{mm}(k)$')

		Pk_bf_mp = np.interp(k_mp, k, Pk_bf[0, 1, 0])
		Pk_default_mp = np.interp(k_mp, k, Pk_default[0, 1, 0])
		plot_Pk(ax[2, 0], ax[3, 0], k_mp, Pk_mp, variance_Pk_mp, Pk_bf_mp, Pk_default_mp, Pk_label_mp)
		plot_Pk(ax[2, 1], ax[3, 1], k_mp, Pk_mp/Pk_mm_dmo, variance_Pk_mp/Pk_mm_dmo**2, 
                Pk_bf_mp/Pk_HMx_dmo, Pk_default_mp/Pk_HMx_dmo, '$10^3R_{mp}(k)$')

		if fit_response: 
			ax[0, 1].set_title('Fit Quantity response R(k)')
		else: 
			ax[0, 0].set_title('Fit Quantity power spectrum P(k)')
		ax[0, 0].set_yscale('log')
		ax[2, 0].set_yscale('log')

	plt.savefig(f'{save_dir}/HMx_bestfit.pdf', dpi=300, bbox_inches='tight')

