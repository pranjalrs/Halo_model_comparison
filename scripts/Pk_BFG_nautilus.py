import corner
import joblib
import matplotlib.pyplot as plt
from mpi4py.futures import MPIPoolExecutor
import numpy as np
import os
import sys
import yaml

import pyccl as ccl
from nautilus import Prior, Sampler

import mcmc
from fit_Pk_BFG import init_config, init_data

import warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"

# config_path = 'Mead20_mm_mp_2.yaml' # sys.argv[1]
config_path = 'Arico20_mm_mp_2.yaml' # sys.argv[1]
# config_path = 'Schneider19_mm_mp_2.yaml' # sys.argv[1]

with open(f'../config/{config_path}', 'r') as stream:
	config = yaml.safe_load(stream)

config = init_config(config, config_path)
config['save_dir'] = f'../BFG_chains_nautilus/{config_path.split(".")[0]}'


ndim = config['ndim']
redshift = 0.25
a = 1/(1+redshift)
# Build cosmology
print('[INFO]: Building cosmology...')
cosmo = ccl.Cosmology(Omega_c=0.2264, Omega_b=0.0456, Omega_g=0, Omega_k=0.0,
				h=0.704, sigma8=0.809, n_s=0.963, Neff=3.04, m_nu=0.0,
				w0=-1, wa=0, transfer_function='boltzmann_camb', extra_parameters={'kmax':200.})

print('[INFO]: Initializing BFG model...')
# 	cosmo = ccl.CosmologyVanillaLCDM()
bfg_dict = mcmc.init_bfg_model(config['halo_model'], cosmo, a=a)

print('[INFO]: Initializing data...')
# Load data
Pk_data = init_data()

mcmc.bfg_dict = bfg_dict
mcmc.Pk_data = Pk_data
mcmc.config = config


if not os.path.isdir(config['save_dir']):
	print(f"Creating directory {config['save_dir']}")
	os.makedirs(config['save_dir'])

#----------------------------------- 4. Setup Nautilus -----------------------------------#
prior = Prior()

for i, this_param in enumerate(config['fit_params']):
	this_prior = config['priors'][i]
	prior.add_parameter(this_param, dist=(this_prior[0], this_prior[1]))

def likelihood(param_dict):
	x = [param_dict[key] for key in config['fit_params']]
	return mcmc.log_likelihood(x, return_total=True, apply_prior=False)


if __name__ == '__main__':
	run = True #bool(int(sys.argv[1]))

	if run:
		sampler = Sampler(prior, likelihood, n_live=3000, filepath=config['save_dir']+'/checkpoint.hdf5', pool=MPIPoolExecutor())
		sampler.run(verbose=True)


		points, log_w, log_l = sampler.posterior()

		# 1. Make corner plot
		corner.corner(points, weights=np.exp(log_w), bins=20, labels=prior.keys, color='purple',
			plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)))
		plt.savefig(f'{config["save_dir"]}/corner.pdf')

		# 2. Save best fit
		bf_params = np.average(points, weights=np.exp(log_w), axis=0)
		fig = mcmc.save_best_fit(bf_params, Pk_data, config, bfg_dict)
		fig.savefig(f'{config["save_dir"]}/best_fit_Pk.pdf', dpi=300, bbox_inches='tight')

		# 3. Save samples
		samples_dict = {'points': points, 'log_w': log_w, 'log_l': log_l}
		joblib.dump(samples_dict, f'{config["save_dir"]}/samples_dict.pkl')

		# 4. Save YAML file
		with open(f'{config["save_dir"]}/config.yaml', 'w') as stream:
			yaml.dump(config, stream)

	else:
		samples_dict = joblib.load(f'{config["save_dir"]}/samples_dict.pkl')
		points = samples_dict['points']
		log_w = samples_dict['log_w']
		log_l = samples_dict['log_l']

		corner.corner(points, weights=np.exp(log_w), bins=20, labels=prior.keys, color='purple',
			plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)))
		plt.savefig(f'{config["save_dir"]}/corner.pdf')