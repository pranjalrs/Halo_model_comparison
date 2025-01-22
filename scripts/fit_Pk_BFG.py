import numpy as np
import sys
import yaml

import BaryonForge as bfg
import pyccl as ccl

from mcmc import run_mcmc_multiprocessing, run_mcmc_MPI, save_summary_plots

def get_Pk_var(Pk, k, boxsize):
	delta_k = 2*np.pi/boxsize
	Nk = 2*np.pi * (k/delta_k)**2
	return Pk**2/Nk

def init_config(config):
	params_dict = config['params']
	fit_params = list(params_dict.keys())
	priors = [params_dict['prior'] for param in fit_params]
	latex_name = [params_dict['latex_name'] for param in fit_params]

	config['initial_guess'] = [params_dict['initial_value'] for param in fit_params]
	config['priors'] = priors
	config['ndim'] = len(fit_params)
	config['latex_name'] = latex_name
	config['fit_params'] = fit_params
	config['save_dir'] = f'../BFG_chains/{config_path.split(".")[0]}'

	return config

class BFGModel:
	def __init__(self, cosmo, a):
		self.k = np.logspace(-2, 1.5, 100)
		self.a = a

		par = bfg.Profiles.Mead20.Params_TAGN_7p8_MPr
		par['M0'] = 10**13.59
		self.rho  = ccl.rho_x(cosmo, 1, 'matter', is_comoving = True)
		self.fft_precision = dict(padding_lo_fftlog = 1e-8, padding_hi_fftlog = 1e8, n_per_decade = 100)

		self.mdef = ccl.halos.MassDef(Delta="vir", rho_type="matter")

		#We will use the built-in, CCL halo model calculation tools.
		hmf = ccl.halos.MassFuncSheth99(mass_def=self.mdef, mass_def_strict = False, use_delta_c_fit = True)
		hbf = ccl.halos.HaloBiasSheth99(mass_def=self.mdef, mass_def_strict = False, use_delta_c_fit = True)
		self.HMC  = ccl.halos.halo_model.HMCalculator(mass_function = hmf, halo_bias = hbf,
												mass_def = self.mdef,
												log10M_min = 8, log10M_max = 16, nM = 100)


		# Compute DMO power spectrum for response calculation
		DMO = bfg.Profiles.Mead20.DarkMatter(mass_def = self.mdef) / self.rho
		DMO.update_precision_fftlog(**self.fft_precision)
		self.P_mm_dmo = ccl.halos.pk_2pt.halomod_power_spectrum(self.cosmo, self.HMC, self.k, self.a, DMO)


	def __call__(self, param_values=None, fit_params=None, field=None):
		par = {key: val for key, val in zip(fit_params, param_values)}
		#Define profiles. Normalize to convert density --> overdensity
		DMB = bfg.Profiles.Mead20.DarkMatterBaryon(**par, mass_def = self.mdef) / self.rho
		PRS = bfg.Profiles.Mead20.Pressure(**par, mass_def =self.mdef)
		Gas = bfg.Profiles.Mead2020.Gas(**par)
		GasDensity = bfg.Profiles.GasNumberDensity(gas = Gas, mean_molecular_weight = 1.15) #simple constant rescaling of gas density --> number density in cgs
		ElectronDensity = GasDensity

		#Upgrade precision of all profiles.
		for p in [DMB, PRS]: p.update_precision_fftlog(**self.fft_precision)
			#Compute all power spectra. This routine has some functionality to mimic the
		#transition-regime modelling of Mead++ in HMCode.
		if field == 'm-m':
			Pk = ccl.halos.pk_2pt.halomod_power_spectrum(self.cosmo, self.HMC, self.k, self.a, DMB)

		elif field == 'm-p':
			Pk = ccl.halos.pk_2pt.halomod_power_spectrum(self.cosmo, self.HMC, self.k, self.a, DMB, prof2 = PRS)

		elif field == 'ne-ne':
			Pk = ccl.halos.pk_2pt.halomod_power_spectrum(self.cosmo, self.HMC, self.k, self.a, ElectronDensity)

		return Pk, self.k


class Likelihood:
	def __init__(self, data, config, bfg_model):
		self.bfg_model = bfg_model
		self.data = data

		# Config
		self.fields = config['fields']
		self.fit_response = config['fit_response']
		self.fit_params = config['fit_params']
		self.priors = config['priors']

		self.response_denom_theory = self.bfg_model.P_mm_dmo if self.fit_response else 1
		self.response_denom_data = data['dmo']['Pk'] if self.fit_response else 1

	def __call__(self, param_values):
		if param_values is not None:
			for i, this_param in enumerate(self.fit_params):
				prior = self.priors[i]
				if not (prior[0] <= param_values[i] <= prior[1]):
					return -np.inf

		log_likelihood = 0.

		for field in self.fields:
			Pk_theory, k = self.bfg_model(param_values, self.fit_params, field)
			Pk_sim = self.data[field]['Pk']/self.response_denom_data
			k_sim = self.data[field]['k']
			variance = self.data[field]['variance']

			Pk_theory_interp = np.interp(k_sim, k, Pk_theory/self.response_denom_theory)

			log_likelihood += -0.5*np.sum((Pk_theory_interp - Pk_sim)**2/variance)


		return log_likelihood


#----------------------------------- 2. Load Data -----------------------------------#
kmax = 6 # h/Mpc

Pk_data = {}
# Load Pk for DMO
k_mm_dmo, Pk_mm_dmo = np.loadtxt('../magneticum-data/data/Pylians/Pk_matter/Box2/Pk_hr_dm_CIC_R1024.txt', unpack=True)
Pk_mm_dmo = Pk_mm_dmo[k_mm_dmo < kmax]
k_mm_dmo = k_mm_dmo[k_mm_dmo < kmax]
Pk_data['dmo'] = {'k': k_mm_dmo, 'Pk': Pk_mm_dmo}

# Load Pk for matter-matter
k_mm, Pk_mm, box_size = np.loadtxt('../magneticum-data/data/Pylians/Pk_matter/Box2/Pk_hr_bao_CIC_R1024.txt', unpack=True), 352
Pk_mm, k_mm = Pk_mm[k_mm < kmax], k_mm[k_mm < kmax]
variance_Pk_mm = get_Pk_var(Pk_mm, k_mm, box_size)

Pk_data['m-m'] = {'k': k_mm, 'Pk': Pk_mm, 'variance': variance_Pk_mm}

# Load Pk for matter-pressure
k_mp, Pk_mp, box_size = np.loadtxt('../magneticum-data/data/Pylians/Pk_matterxpressure/Box2_CIC_R1024.txt', unpack=True), 352

Pk_mp = Pk_mp[:, 1]*1e3  # Convert to eV/cm^3
Pk_mp,	k_mp = Pk_mp[k_mp < kmax], k_mp[k_mp < kmax]
variance_Pk_mp = get_Pk_var(Pk_mp, k_mp, box_size)

Pk_data['m-p'] = {'k': k_mp, 'Pk': Pk_mp, 'variance': variance_Pk_mp}

# Load Pk for electron density
# Check that all Pk are sampled on the same grid
k_data = [Pk_data[key]['k'] for key in Pk_data.keys()]
assert all([np.all(k_data[0] == k) for k in k_data]), 'The Pk are not sampled on the same grid'


if __name__ == '__main__':
	config_path = sys.argv[1]

	with open(f'../config/{config_path}', 'r') as stream:
		config = yaml.safe_load(stream)

	config = init_config(config)
	burnin = config['mcmc']['burnin']
	parallel = config['mcmc']['parallel']
	ndim = config['ndim']

	likelihood = Likelihood(Pk_data, config)

		#----------------------------------- 4. Run emcee -----------------------------------#

	if parallel == 'multiprocessing':
		sampler, walkers, flat_chain = run_mcmc_multiprocessing(likelihood, config)

	elif parallel == 'MPI':
		sampler, walkers, flat_chain = run_mcmc_MPI(likelihood, config)
		np.save(f'{config['save_dir']}/samples.npy', walkers)

	if parallel == 'load_samples':
		walkers = np.load(f'{config['save_dir']}/samples.npy')
		nsteps, nwalkers, ndim = walkers.shape
		# Discard 80% and flatten chain
		flat_chain = walkers[int(burnin*nsteps):, :, :].reshape((int((1-burnin)*nsteps)*nwalkers, ndim))


	save_summary_plots(walkers, flat_chain, config)