import corner
import emcee
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from schwimmbad import MPIPool
import sys


def run_mcmc_multiprocessing(likelihood, config):
	burnin = config['mcmc']['burnin']
	nwalkers = config['mcmc']['nwalkers']
	nsteps = config['mcmc']['nsteps']
	ndim = config['ndim']
	save_dir = config['save_dir']

	initial_guess = np.array(config['initial_guess']) + 1e-3*np.random.randn(nwalkers, ndim)

	with Pool() as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, pool=pool)
		sampler.run_mcmc(initial_guess, nsteps, progress=True)
		walkers = sampler.get_chain(flat=False)
		np.save(f'{save_dir}/samples.npy', walkers)
		flat_chain = sampler.get_chain(discard=int(burnin*nsteps), flat=True)

	return sampler, walkers, flat_chain


def run_mcmc_MPI(likelihood, config, save_dir):
	burnin = config['mcmc']['burnin']
	nwalkers = config['mcmc']['nwalkers']
	nsteps = config['mcmc']['nsteps']
	niter = config['mcmc']['niter']
	ndim = config['ndim']

	walkers = []
	with MPIPool() as pool:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)
		for i in range(niter):
			print('Iteration: ', i+1)
			print('\n')
			sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, pool=pool)
			sampler.run_mcmc(initial_guess, 300, progress=True)
			walkers.append(sampler.get_chain(flat=False))
			flat_chain = sampler.get_chain(flat=True)
			blobs = sampler.get_blobs(flat=True)
			idx = np.argmax(blobs)
			initial_guess = np.array(flat_chain[idx]) + 1e-3*np.random.randn(nwalkers, ndim)
			print('\n')

	# Now run long chain
		print('Final iteration')
		sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, pool=pool)
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