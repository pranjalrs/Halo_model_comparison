import numpy as np
import os
import sys
import yaml
import emcee

from multiprocessing import Pool
import pyccl as ccl

import mcmc

def get_Pk_var(Pk, k, boxsize):
	delta_k = 2*np.pi/boxsize
	Nk = 2*np.pi * (k/delta_k)**2
	return Pk**2/Nk


def init_config(config):
	params_dict = config['params']
	fit_params = list(params_dict.keys())
	priors = [params_dict[param]['prior'] for param in fit_params]
	latex_name = [params_dict[param]['latex_name'] for param in fit_params]
	config['initial_guess'] = [params_dict[param]['initial_value'] for param in fit_params]

	config['priors'] = priors
	config['ndim'] = len(fit_params)
	config['latex_names'] = latex_name
	config['fit_params'] = fit_params
	config['save_dir'] = f'../BFG_chains/{config_path.split(".")[0]}'

	return config

def init_data():
	print('Loading Pk data...')
	kmax = 6 # h/Mpc

	Pk_data = {}
	# Load Pk for DMO
	(k_mm_dmo, Pk_mm_dmo) = np.loadtxt('../data/Pk_m-m/Pk_Box2hr_dm_CIC_R1024.txt', unpack=True)
	Pk_mm_dmo = Pk_mm_dmo[k_mm_dmo < kmax]
	k_mm_dmo = k_mm_dmo[k_mm_dmo < kmax]
	Pk_data['dmo'] = {'k': k_mm_dmo, 'Pk': Pk_mm_dmo}

	# Load Pk for matter-matter
	(k_mm, Pk_mm), box_size = np.loadtxt('../data/Pk_m-m/Pk_Box2hr_bao_CIC_R1024.txt', unpack=True), 352
	Pk_mm, k_mm = Pk_mm[k_mm < kmax], k_mm[k_mm < kmax]
	variance_Pk_mm = get_Pk_var(Pk_mm, k_mm, box_size)

	Pk_data['m-m'] = {'k': k_mm, 'Pk': Pk_mm, 'variance': variance_Pk_mm}

	# Load Pk for matter-pressure
	(k_mp, Pk_mp), box_size = np.loadtxt('../data/Pk_m-p/Pk_Box2hr_CIC_R1024.txt', unpack=True), 352

	Pk_mp = Pk_mp*1e3  # Convert to eV/cm^3
	Pk_mp, k_mp = Pk_mp[k_mp < kmax], k_mp[k_mp < kmax]
	variance_Pk_mp = get_Pk_var(Pk_mp, k_mp, box_size)

	Pk_data['m-p'] = {'k': k_mp, 'Pk': Pk_mp, 'variance': variance_Pk_mp}

	# Load Pk for electron density
	# Pk_ne in h^2/cm^3
	(k_ne, Pk_ne), box_size = np.loadtxt('../data/Pk_ne-ne/Pk_Box2hr_ne_Mead_R1024.txt', unpack=True), 352
	Pk_ne, k_ne = Pk_ne[k_ne < kmax], k_ne[k_ne < kmax]
	variance_Pk_ne = get_Pk_var(Pk_ne, k_ne, box_size)

	Pk_data['ne-ne'] = {'k': k_ne, 'Pk': Pk_ne, 'variance': variance_Pk_ne}

	# Check that all Pk are sampled on the same grid
	k_data = [Pk_data[key]['k'] for key in Pk_data.keys()]
	assert all([np.all(k_data[0] == k) for k in k_data]), 'The Pk are not sampled on the same grid'
	return Pk_data

if __name__ == '__main__':
	config_path = sys.argv[1]

	with open(f'../config/{config_path}', 'r') as stream:
		config = yaml.safe_load(stream)

	config = init_config(config)
	burnin = config['mcmc']['burnin']
	parallel = config['mcmc']['parallel']
	ndim = config['ndim']

	# Build cosmology
	print('Building cosmology...')
	cosmo = ccl.Cosmology(Omega_c=0.2264, Omega_b=0.0456, Omega_g=0, Omega_k=0.0,
					h=0.704, sigma8=0.809, n_s=0.963, Neff=3.04, m_nu=0.0,
					w0=-1, wa=0, transfer_function='boltzmann_camb', extra_parameters={'kmax':200.})

# 	cosmo = ccl.CosmologyVanillaLCDM()
	bfg_dict = mcmc.init_bfg_model(config['halo_model'], cosmo, a=1)

	# Load data
	Pk_data = init_data()

	mcmc.bfg_dict = bfg_dict
	mcmc.Pk_data = Pk_data
	mcmc.config = config

	if not os.path.isdir(config['save_dir']):
		print(f"Creating directory {config['save_dir']}")
		os.makedirs(config['save_dir'])

	#----------------------------------- 4. Run emcee -----------------------------------#
	if parallel == 'test':
		sampler, walkers, flat_chain = mcmc.run_mcmc(config)

	elif parallel == 'multiprocessing':
		sampler, walkers, flat_chain = mcmc.run_mcmc_multiprocessing(config)

	elif parallel == 'MPI':
		sampler, walkers, flat_chain = mcmc.run_mcmc_MPI(config)
		np.save(f'{config["save_dir"]}/samples.npy', walkers)

	elif parallel == 'load_samples':
		walkers = np.load(f'{config["save_dir"]}/samples.npy')
		nsteps, nwalkers, ndim = walkers.shape
		flat_chain = walkers[int(burnin*nsteps):, :, :].reshape((nsteps-int(burnin*nsteps))*nwalkers, ndim)
		# flat_chain = walkers.reshape(nsteps*nwalkers, ndim)

	mcmc.save_summary_plots(walkers, flat_chain, config)
	mcmc.save_best_fit(flat_chain, Pk_data, config, bfg_dict)