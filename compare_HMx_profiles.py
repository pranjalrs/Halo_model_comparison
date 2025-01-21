import numpy as np
import matplotlib.gridspec as gridspec
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
		 'alpha': 0.8471,              
		 'omega_m': 0.272,
		 'omega_b': 0.0456}
		return params_default
    
	if 'eps_2' in param_names:
		eps_2 = param_values[param_names.index('eps_2')]
    
	else: eps_2 = 0

	if 'alpha' in param_names:
		alpha = param_values[param_names.index('alpha')]
    
	else: alpha = 0.8471

	params = {'eps1_0': param_values[param_names.index('eps_1')],
			'eps1_1': 0, # not relevant, fitting single redshift
			'eps2_0': eps_2,  # default HMx value
			'eps2_1': 0,  # not relevant, fitting single redshift
			'beta': 0.6, # Mass scaling of bnd gas fraction; default HMx value,
			'M0': 10**param_values[param_names.index('logM0')]*u.Msun/cu.littleh,
			'gamma': param_values[param_names.index('gamma')],
			'alpha': alpha,
			'omega_m': 0.272,
			'omega_b': 0.0456}

	return params

def plot_profiles(ax, nrow, ncol, mass_bin_edges, mean_profiles, HMx_profs_mm, HMx_profs_joint, HMx_profs_default, mean_mvirs, Nhalos, field=None):


	for i in range(len(mass_bin_edges)-1):
		r = mean_profiles_gas[i][1]
		x = mean_profiles_gas[i][2]  # r/rvir
		#---------------------------------- Plot gas profiles -------------------------------#
		ax[i][0].errorbar(r/1e3, 4*np.pi*r**2*mean_profiles[i][0], yerr=4*np.pi*r**2*mean_profiles[i][-1], capsize=3, ls='', fmt='^', c='k', label='Magneticum')
		ax[i][0].errorbar(r/1e3, 4*np.pi*r**2*mean_profiles[i][0], yerr=4*np.pi*r**2*mean_profiles[i][-1], capsize=3, ls='', fmt='^', c='k', label='Magneticum')

		ax[i][0].loglog(r/1e3, 4*np.pi*r**2*HMx_profs_default[i].value, '.-', c='limegreen', lw=1.2, label='HMx default')
		ax[i][0].loglog(r/1e3, 4*np.pi*r**2*HMx_profs_mm[i].value, '--', c='orangered', lw=1.2, label='HMx fit (mm)')
		ax[i][0].loglog(r/1e3, 4*np.pi*r**2*HMx_profs_joint[i].value, '--', c='cornflowerblue', lw=1.2, label='HMx fit (mm+mp)')

		ax[i][1].semilogx(r/1e3, HMx_profs_mm[i].value/mean_profiles[i][0], '--', c='orangered', lw=1.2, label='gas: HMx fit')
		ax[i][1].semilogx(r/1e3, HMx_profs_joint[i].value/mean_profiles[i][0], '--', c='cornflowerblue', lw=1.2, label='gas: HMx fit')
		ax[i][1].semilogx(r/1e3, HMx_profs_default[i].value/mean_profiles[i][0], '.-', c='limegreen', lw=1.2, label='gas: HMx default')
		ax[i][1].axhline(1, c='gray', ls=':')

		log_mmin, log_mmax = np.log10(mass_bin_edges[i]), np.log10(mass_bin_edges[i+1])

		profiles = [mean_profiles[i][0], HMx_profs_mm[i].value, HMx_profs_joint[i].value, HMx_profs_default[i].value]
		mask = r>=20  # in kpc/h
		y_min = 4*np.pi*min(min(r[mask]**2*profile[mask]) for profile in profiles)
		y_max = 4*np.pi*max(max(r[mask]**2*profile[mask]) for profile in profiles)
		ax[i][0].set_ylim(0.95*y_min, 1.25*y_max)

		y_min = min(min(profile[mask]/mean_profiles[i][0][mask]) for profile in profiles)
		y_max = max(max(profile[mask]/mean_profiles[i][0][mask]) for profile in profiles)    
		ax[i][1].set_ylim(0.95*y_min, 1.05*y_max)

		ax[i][0].set_xlim(20/1e3)
		ax[i][1].set_xlim(20/1e3)

		ax[i][1].set_xlabel('$r$ [Mpc/$h$]')
		ax[i][0].text(0.5, 0.25, f'{log_mmin:.2f}<logM<{log_mmax:.2f}\nNhalos={Nhalos[i]}', transform=ax[i][0].transAxes)

        
		ax1 = ax[i][0].twiny()
		ax1.plot(x, r**2*mean_profiles[i][0], ls="", color="none")  # dummy trace
# 		ax1.xaxis.set_major_locator(ax[idx].xaxis.get_major_locator())
# 		ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{k*x:g}"))
		ax1.set_xlabel("$r/R_\mathrm{virial}$")

	ylabel1 = '$4\pi r^2P_e [\mathrm{eV}\,\mathrm{kpc}^2 \mathrm{cm}^{-3}]$' if field == 'Pe' else '$4\pi r^2\\rho [\mathrm{M}_\odot\mathrm{kpc}^{-1}]$'
	ylabel2 = '$P_\mathrm{e, HMx}/P_\mathrm{e, sim}$' if field=='Pe' else '$\\rho_\mathrm{HMx}/\\rho_\mathrm{sim}$' 

	for i in [0]:
		idx = 2*i - i%ncol
		ax[i][0].set_ylabel(ylabel1)
		ax[i][1].set_ylabel(ylabel2)


def build_axes(nrow, ncol, height_ratios):
    fig = plt.figure(figsize=(ncol*5.5, nrow*5.5))
    subfig = fig.subfigures(nrow, ncol, wspace=-0.4)
    subfig = subfig.flatten()

    color = ['red', 'blue', 'orange', 'k', 'yellow', 'violet']
    all_axes = []
    for i in range(len(subfig)):
#         subfig[i].set_facecolor(color[i])
        ax = subfig[i].subplots(2, 1, gridspec_kw={'height_ratios': height_ratios, 'hspace':0.})

        all_axes.append(ax)

    return fig, all_axes
#------------------------------------ 1. Load chain and get best fit params ------------------------------------#
# Chain for matter-matter fit
chain = np.load('chains/Pk_mm_fit_nparam_3_fit_response/HMx.npy')
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
m_min, m_max, n_bins = 12, 15, 3
mass_bin_edges = np.logspace(m_min, m_max, n_bins+1)
_, mean_profiles_cdm, mean_profiles_gas, mean_profiles_Pe, _, _, mean_mvirs, Nhalos = simulation_profiles.get_binned_profiles(m_min, m_max, n_bins)

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
    this_gas_prof = this_gas_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_gas_profs_mm.append(this_gas_prof)

    this_gas_prof, _ = HMx_profiles.get_rho_gas_profile(this_mvir, z=0, params=best_fit_param_joint, x_bins=x_bins)
    this_gas_prof = this_gas_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_gas_profs_joint.append(this_gas_prof)


    # Also get profile for default parameters
    this_gas_prof, _ = HMx_profiles.get_rho_gas_profile(this_mvir, z=0, params=param_default, x_bins=x_bins)
    this_gas_prof = this_gas_prof.to(u.Msun*cu.littleh**2/u.kpc**3)
    HMx_gas_profs_default.append(this_gas_prof)


HMx_Pe_profs_mm = []
HMx_Pe_profs_joint = []
HMx_Pe_profs_default = []

for i in range(len(mass_bin_edges)-1):
    this_mvir = mean_mvirs[i]*u.Msun/cu.littleh
    x_bins = mean_profiles_cdm[i][2]

    this_Pe_prof, _ = HMx_profiles.get_Pe_profile(this_mvir, z=0, params=best_fit_param_mm, x_bins=x_bins)
    this_Pe_prof = this_Pe_prof.to(u.eV*cu.littleh**2/u.cm**3)
    HMx_Pe_profs_mm.append(this_Pe_prof)

    this_Pe_prof, _ = HMx_profiles.get_Pe_profile(this_mvir, z=0, params=best_fit_param_joint, x_bins=x_bins)
    this_Pe_prof = this_Pe_prof.to(u.eV*cu.littleh**2/u.cm**3)
    HMx_Pe_profs_joint.append(this_Pe_prof)


    # Also get profile for default parameters
    this_Pe_prof, _ = HMx_profiles.get_Pe_profile(this_mvir, z=0, params=param_default, x_bins=x_bins)
    this_Pe_prof = this_Pe_prof.to(u.eV*cu.littleh**2/u.cm**3)
    HMx_Pe_profs_default.append(this_Pe_prof)

#------------------------------------ 4. Plot profiles ------------------------------------#
ncol, nrow = 3, 1
assert ncol*nrow >= n_bins, 'Increase number of columns or rows'

#####------------------- 4a. CDM profiles -------------------#####
# fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*4.5, nrow/2*4.5), sharex=False, gridspec_kw={'height_ratios': height_ratios, 'hspace':0.4})

fig, ax = build_axes(nrow, ncol, [3, 1])
# ax = ax.flatten()

# Plot cdm profiles
plot_profiles(ax, nrow, ncol, mass_bin_edges, mean_profiles_cdm, HMx_cdm_profs_mm, HMx_cdm_profs_joint, HMx_cdm_profs_default, mean_mvirs, Nhalos)
ax[0][0].legend(ax[0][0].get_legend_handles_labels()[0][:2], ax[0][0].get_legend_handles_labels()[1][:2], loc='upper right',  title='CDM profiles')
ax[1][0].legend(ax[0][0].get_legend_handles_labels()[0][2:], ax[0][0].get_legend_handles_labels()[1][2:], loc='upper right',  title='CDM profiles')

plt.savefig('chains/cdm_profiles_comparison.pdf')


#####------------------- 4b. Gas profiles -------------------#####
fig, ax = build_axes(nrow, ncol, [3, 1])


plot_profiles(ax, nrow, ncol, mass_bin_edges, mean_profiles_gas, HMx_gas_profs_mm, HMx_gas_profs_joint, HMx_gas_profs_default, mean_mvirs, Nhalos)
ax[0][0].legend(ax[0][0].get_legend_handles_labels()[0][:2], ax[0][0].get_legend_handles_labels()[1][:2], loc='upper right',  title='Gas profiles')
ax[1][0].legend(ax[0][0].get_legend_handles_labels()[0][2:], ax[0][0].get_legend_handles_labels()[1][2:], loc='upper right',  title='Gas profiles')
plt.savefig('chains/gas_profiles_comparison.pdf')


#####------------------- 4c. Pressure profiles -------------------#####
fig, ax = build_axes(nrow, ncol, [3, 1])


plot_profiles(ax, nrow, ncol, mass_bin_edges, mean_profiles_Pe, HMx_Pe_profs_mm, HMx_Pe_profs_joint, HMx_Pe_profs_default, mean_mvirs, Nhalos, 'Pe')
ax[0][0].legend(ax[0][0].get_legend_handles_labels()[0][:2], ax[0][0].get_legend_handles_labels()[1][:2], loc='upper right',  title='$P_e$ profiles')
ax[1][0].legend(ax[0][0].get_legend_handles_labels()[0][2:], ax[0][0].get_legend_handles_labels()[1][2:], loc='upper right',  title='$P_e$ profiles')
plt.savefig('chains/Pe_profiles_comparison.pdf')