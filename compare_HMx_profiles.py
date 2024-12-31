import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.cosmology.units as cu

import sys
sys.path.append('./src/')

import HMx_profiles
import simulation_profiles


def get_param_dict(param_values=None, param_names=None):
	if param_values is None:
		params_default = {'eps1_0': -0.1065,
		  'eps1_1': 0, # not relevant, fitting single redshift
		 'eps2_0': 0.,  # default HMx value
		 'eps2_1': 0,  # not relevant, fitting single redshift
		 'beta': 0.6, # Mass scaling of bnd gas fraction; default HMx value,
		 'M0': 10**13.5937*u.Msun/cu.littleh,
		 'gamma': 1.177,
		 'omega_m': 0.272,
		 'omega_b': 0.0456}
		return params_default

	params = {'eps1_0': param_values[param_names.index('eps_1')],
			'eps1_1': 0, # not relevant, fitting single redshift
			'eps2_0': param_values[param_names.index('eps_2')],  # default HMx value
			'eps2_1': 0,  # not relevant, fitting single redshift
			'beta': 0.6, # Mass scaling of bnd gas fraction; default HMx value,
			'M0': 10**param_values[param_names.index('logM0')]*u.Msun/cu.littleh,
			'gamma': param_values[param_names.index('gamma')],
			'omega_m': 0.272,
			'omega_b': 0.0456}

	return params

def plot_profiles(ax, nrow, ncol, mass_bin_edges, mean_profiles, HMx_profs_mm, HMx_profs_joint, HMx_profs_default, mean_mvirs, Nhalos):

	for i in range(len(mass_bin_edges)-1):
		idx = 2*i - i%ncol
		r = mean_profiles_gas[i][1]/1e3
		#---------------------------------- Plot gas profiles -------------------------------#
		ax[idx].scatter(r, r**2*mean_profiles[i][0], '^', c='k', label='Gas: Magneticum')

		ax[idx].loglog(r, r**2*HMx_profs_mm[i].value, '--', c='orangered', lw=1.2, label='Gas: HMx fit (mm)')
		ax[idx].loglog(r, r**2*HMx_profs_joint[i].value, '--', c='cornflowerblue', lw=1.2, label='Gas: HMx fit (mm + mp)')
		ax[idx].loglog(r, r**2*HMx_profs_default[i].value, '.-', c='limegreen', lw=1.2, label='Gas: HMx default')

		ax[idx+ncol].semilogx(r, HMx_profs_mm[i].value/mean_profiles[i][0], '--', c='orangered', lw=1.2, label='gas: HMx fit')
		ax[idx+ncol].semilogx(r, HMx_profs_joint[i].value/mean_profiles[i][0], '--', c='cornflowerblue', lw=1.2, label='gas: HMx fit')
		ax[idx+ncol].semilogx(r, HMx_profs_default[i].value/mean_profiles[i][0], '.-', c='limegreen', lw=1.2, label='gas: HMx default')
		ax[idx+ncol].axhline(1, c='gray', ls=':')

		log_mmin, log_mmax = np.log10(mass_bin_edges[i]), np.log10(mass_bin_edges[i+1])
		ax[idx].set_xlabel('r [Mpc/h]')
		ax[idx].text(0.5, 0.85, f'{log_mmin:.2f}<logM<{log_mmax:.2f}\nNhalos={Nhalos[i]}', transform=ax[idx].transAxes)


		ax[0].legend(frameon=False, loc='lower left')


	for i in [0, 3]:
		idx = 2*i - i%ncol
		ax[idx].set_ylabel('$r^2\\rho [\mathrm{M}_\odot\mathrm{kpc}^{-1}]$')
		ax[idx+ncol].set_ylabel('$\\rho_\mathrm{HMx}/\\rho_\mathrm{sim}$')

	plt.tight_layout()

#------------------------------------ 1. Load chain and get best fit params ------------------------------------#
# Chain for matter-matter fit
chain = np.load('chains/Pk_mm_fit_nparam_3/HMx.npy')
nsteps, nwalkers, nparam = chain.shape
burn_in = int(nsteps*0.7)
flat_chain = chain[burn_in:, :, :].reshape(nwalkers*(nsteps-burn_in), nparam)
fit_params = ['eps_1', 'gamma', 'logM0']

best_fit_mm = np.mean(flat_chain, axis=0)
best_fit_param_mm = get_param_dict(best_fit_mm, fit_params)

# Chain for joint fit
chain = np.load('chains/Pk_joint_fit_nparam_6_fit_response/HMx.npy')
nsteps, nwalkers, nparam = chain.shape
burn_in = int(nsteps*0.7)
flat_chain = chain[burn_in:, :, :].reshape(nwalkers*(nsteps-burn_in), nparam)
fit_params = ['eps_1', 'gamma', 'logM0', 'alpha', 'twhim', 'eps_2']

best_fit_joint = np.mean(flat_chain, axis=0)
best_fit_param_joint = get_param_dict(best_fit_joint, fit_params)

param_default = get_param_dict()
#------------------------------------ 2. Get binned profiles ------------------------------------#
m_min, m_max, n_bins = 12, 15, 6
mass_bin_edges = np.logspace(m_min, m_max, n_bins+1)
_, mean_profiles_cdm, mean_profiles_gas, _, _, mean_mvirs, Nhalos = simulation_profiles.get_binned_profiles(m_min, m_max, n_bins)

#------------------------------------ 3. Get HMx profiles ------------------------------------#
## Compute analytic CDM profiles
HMx_cdm_profs_mm = []
HMx_cdm_profs_joint = []
HMx_cdm_profs_default = []

for i in range(len(mass_bin_edges)-1):
    this_mvir = mean_mvirs[i]*u.Msun/cu.littleh
    x_bins = mean_profiles_cdm[i][2]

    this_cdm_prof, _ = HMx_profiles.get_rho_dm_profile(this_mvir, z=0, params=best_fit_param_mm, x_bins=x_bins)
    this_cdm_prof = this_cdm_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_cdm_profs_mm.append(this_cdm_prof)

    this_cdm_prof, _ = HMx_profiles.get_rho_dm_profile(this_mvir, z=0, params=best_fit_param_joint, x_bins=x_bins)
    this_cdm_prof = this_cdm_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_cdm_profs_joint.append(this_cdm_prof)


    # Also get profile for default parameters
    this_cdm_prof, _ = HMx_profiles.get_rho_dm_profile(this_mvir, z=0, params=param_default, x_bins=x_bins)
    this_cdm_prof = this_cdm_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_cdm_profs_default.append(this_cdm_prof)


HMx_gas_profs_mm = []
HMx_gas_profs_joint = []
HMx_gas_profs_default = []

for i in range(len(mass_bin_edges)-1):
    this_mvir = mean_mvirs[i]*u.Msun/cu.littleh
    x_bins = mean_profiles_cdm[i][2]

    this_gas_prof, _ = HMx_profiles.get_rho_gas_profile(this_mvir, z=0, params=best_fit_param_mm, x_bins=x_bins)
    this_gas_prof = this_cdm_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_gas_profs_mm.append(this_gas_prof)

    this_gas_prof, _ = HMx_profiles.get_rho_gas_profile(this_mvir, z=0, params=best_fit_param_joint, x_bins=x_bins)
    this_gas_prof = this_cdm_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_gas_profs_joint.append(this_gas_prof)


    # Also get profile for default parameters
    this_gas_prof, _ = HMx_profiles.get_rho_gas_profile(this_mvir, z=0, params=param_default, x_bins=x_bins)
    this_gas_prof = this_gas_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_gas_profs_default.append(this_gas_prof)


#------------------------------------ 4. Plot profiles ------------------------------------#
ncol, nrow = 3, 4
assert ncol*nrow >= n_bins, 'Increase number of columns or rows'

height_ratios = [3, 1]*int(nrow/2)

fig, ax = plt.subplots(nrow, ncol, figsize=(nrow/2*5, ncol*4.5), sharex=False, gridspec_kw={'height_ratios': height_ratios})
ax = ax.flatten()

# Plot cdm profiles
plot_profiles(ax, nrow, ncol, mass_bin_edges, mean_profiles_cdm, HMx_cdm_profs_mm, HMx_cdm_profs_joint, HMx_cdm_profs_default, mean_mvirs, Nhalos)
plt.savefig('chains/cdm_profiles_comparison.pdf')



fig, ax = plt.subplots(nrow, ncol, figsize=(nrow/2*5, ncol*4.5), sharex=False, gridspec_kw={'height_ratios': height_ratios})
ax = ax.flatten()

plot_profiles(ax, nrow, ncol, mass_bin_edges, mean_profiles_gas, HMx_gas_profs_mm, HMx_gas_profs_joint, HMx_gas_profs_default, mean_mvirs, Nhalos)
plt.savefig('chains/gas_profiles_comparison.pdf')