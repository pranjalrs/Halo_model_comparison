import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import yaml

import pyccl as ccl
import BaryonForge as bfg

sys.path.append('../scripts/')
import utils

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
# Axes options
matplotlib.rcParams['axes.titlesize'] = 'x-large'
matplotlib.rcParams['axes.labelsize'] = 'x-large'
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.grid'] = False
#
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['savefig.dpi'] = 300
plt.rc('text', usetex=True)

###------------------------------------------ Plot settings ------------------------------------------###
c_gas = 'dodgerblue'
c_stars = 'coral'
###------------------------------------------ Cosmo ------------------------------------------###

cosmo = ccl.Cosmology(Omega_c=0.2264, Omega_b=0.0456, Omega_g=0, Omega_k=0.0,
				h=0.704, sigma8=0.809, n_s=0.963, Neff=3.04, m_nu=0.0,
				w0=-1, wa=0, transfer_function='boltzmann_camb', extra_parameters={'kmax':200.})
a    = 1/(1+0.25) #Compute everything at z = 0
k    = np.geomspace(1e-3, 20, 100) #Compute across a wide range in k [1/Mpc, comoving]
rho  = ccl.rho_x(cosmo, a, 'matter', is_comoving = True)
h = cosmo.cosmo.params.h
f_b = cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m

mdef_virial = ccl.halos.MassDef(Delta="vir", rho_type="matter")
mdef_200c  = ccl.halos.massdef.MassDef(200, 'critical')
cnvrt = ccl.halos.mass_translator(mass_in = mdef_virial, mass_out = mdef_200c, concentration = 'Duffy08')


Mvir = np.logspace(10, np.log10(2e15), 100)
M200c = np.array([cnvrt(cosmo, M, a) for M in Mvir])

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

###------------------------------------------ Fractions for Mead20 ------------------------------------------###
load_dir = '/groups/timeifler/pranjalrs/Halo_model_comparison/BFG_chains_psobacco/Mead20_mm_mp_2'
samples_dict = joblib.load(f'{load_dir}/backup.pkl')
bf_params = samples_dict['swarm']['pos_bglobal']
with open(f'../config/Mead20_mm_mp_2.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

fit_params = list(config['params'].keys())
par = utils.get_param_dict(bf_params, fit_params, param_dict=None, halo_model='Mead20')

DMB = bfg.Profiles.Mead20.DarkMatterBaryon(**par) / rho

M20_fbnd = np.zeros_like(Mvir)
M20_fej = np.zeros_like(Mvir)
M20_fsga = np.zeros_like(Mvir)
M20_fcga = np.zeros_like(Mvir)

for i, mass in enumerate(Mvir):
	M20_fbnd[i], M20_fej[i] = DMB._get_gas_frac(mass, a, cosmo)
	_, M20_fcga[i], M20_fsga[i] = DMB._get_star_frac(mass, a, cosmo)

ax[0].plot(M200c, M20_fbnd + M20_fej, c=c_gas, label='Total gas')
ax[0].plot(M200c, M20_fbnd, ls='--', c=c_gas, label='Bound gas')
ax[0].plot(M200c, M20_fej, ls='-.', c=c_gas, label='Ejected gas')

ax[0].plot(M200c, M20_fcga + M20_fsga, c=c_stars, label='Total stars')
ax[0].plot(M200c, M20_fcga, ls='--', c=c_stars, label='Central stars')
ax[0].plot(M200c, M20_fsga, ls='-.', c=c_stars, label='Satellite stars')
ax[0].text(0.05, 0.85, 'HMx', c='goldenrod', fontsize=12, weight='semibold', transform=ax[0].transAxes)

###------------------------------------------ Fractions for Schneider19 ------------------------------------------###
load_dir = '/groups/timeifler/pranjalrs/Halo_model_comparison/BFG_chains_psobacco/Schneider19_mm_mp_2'
samples_dict = joblib.load(f'{load_dir}/backup.pkl')
bf_params = samples_dict['swarm']['pos_bglobal']
with open(f'../config/Schneider19_mm_mp_2.yaml', 'r') as stream:
    config = yaml.safe_load(stream)
fit_params = list(config['params'].keys())
bpar_S19 = utils.get_param_dict(bf_params, fit_params, param_dict=None, halo_model='Schneider19')


S19_fbnd = np.zeros_like(M200c)
S19_fcga = np.zeros_like(M200c)
S19_fsga = np.zeros_like(M200c)

for i, mass in enumerate(M200c):
	f_star = 2 * bpar_S19['A'] * ((mass/bpar_S19['M1'])**bpar_S19['tau'] + (mass/bpar_S19['M1'])**bpar_S19['eta'])**-1
	f_gas = f_b - f_star

	eta_cga = bpar_S19['eta'] + bpar_S19['eta_delta']
	tau_cga = bpar_S19['tau'] + bpar_S19['tau_delta']
	f_cga  = 2 * bpar_S19['A'] * ((mass/bpar_S19['M1'])**tau_cga  + (mass/bpar_S19['M1'])**eta_cga)**-1

	f_sga = f_star - f_cga

	S19_fbnd[i] = f_gas
	S19_fcga[i] = f_cga
	S19_fsga[i] = f_sga


ax[1].plot(M200c, S19_fbnd, c=c_gas, label='Bound gas')
ax[1].plot(M200c, S19_fcga + S19_fsga, c=c_stars, label='Total stars')
ax[1].plot(M200c, S19_fcga, ls='--', c=c_stars, label='Central stars')
ax[1].plot(M200c, S19_fsga, ls='-.', c=c_stars, label='Satellite stars')
ax[1].text(0.05, 0.85, 'GODMAX', c='goldenrod', fontsize=12, weight='semibold', transform=ax[1].transAxes)


###------------------------------------------ Fractions for Arico20 ------------------------------------------###
load_dir = '/groups/timeifler/pranjalrs/Halo_model_comparison/BFG_chains_psobacco/Arico20_mm_mp_2'
samples_dict = joblib.load(f'{load_dir}/backup.pkl')
bf_params = samples_dict['swarm']['pos_bglobal']
with open(f'../config/Arico20_mm_mp_2.yaml', 'r') as stream:
    config = yaml.safe_load(stream)
fit_params = list(config['params'].keys())
bpar_A20 = utils.get_param_dict(bf_params, fit_params, param_dict=None, halo_model='Arico20')


BND = bfg.Profiles.Arico20.BoundGas(**bpar_A20) / rho


A20_fbnd = np.zeros_like(Mvir)
A20_fej = np.zeros_like(Mvir)
A20_fsga = np.zeros_like(Mvir)
A20_fcga = np.zeros_like(Mvir)


for i, mass in enumerate(M200c):
	A20_fcga[i] = BND._get_star_frac([mass], a=a, cosmo=cosmo)[0][0]
	A20_fsga[i] = BND._get_star_frac([mass], a=a, cosmo=cosmo, satellite=True)[0][0]

	f_gas = f_b - A20_fcga[i] - A20_fsga[i]
	f_hg  = f_gas / (1 + np.power(bpar_A20['M_c']/mass, bpar_A20['beta']))
	f_rg  = (f_gas - f_hg) / (1 + np.power(bpar_A20['M_r']/mass, bpar_A20['beta_r']))
	f_rg  = np.clip(f_rg, None, f_hg)
	f_bg  = f_hg - f_rg
	f_eg  = f_gas - f_hg

	A20_fbnd[i] = f_hg
	A20_fej[i] = f_eg



ax[2].plot(M200c, A20_fbnd + A20_fej, c=c_gas, label='Total gas')
ax[2].plot(M200c, A20_fbnd, ls='--', c=c_gas, label='Bound gas')
ax[2].plot(M200c, A20_fej, ls='-.', c=c_gas, label='Ejected gas')

ax[2].plot(M200c, A20_fcga + A20_fsga, c=c_stars, label='Total stars')
ax[2].plot(M200c, A20_fcga, ls='--', c=c_stars, label='Central stars')
ax[2].plot(M200c, A20_fsga, ls='-.', c=c_stars, label='Satellite stars')
ax[2].text(0.05, 0.85, 'BCM', c='goldenrod', fontsize=12, weight='semibold', transform=ax[2].transAxes)

# line.set_dashes((5, 2))



###------------------------------------------ Settings for all axes ------------------------------------------###

for subax in ax:
	subax.axhline(1 - f_b, color='purple', label='CDM')
	subax.axhline(f_b, ls='--', color='k', label='$\Omega_\mathrm{b}/\Omega_\mathrm{m}$')

	subax.set_ylim(5e-4, 1.5)
	subax.set_xlim(1e10)
	subax.set_yscale('log')
	subax.set_xscale('log')
	subax.set_xlabel('$M_\mathrm{200c} [\mathrm{M}_\odot/h]$')  # Uncommenting this line to set the x-axis label

	subax.tick_params(axis='both', which='both', direction='in', right=True, top=True)
	subax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))

ax[0].set_ylabel('Mass Fraction')

# Split legend into 3 subplots
handles, labels = ax[0].get_legend_handles_labels()
leg1 = ax[0].legend(handles[-2:], labels[-2:], fontsize=10, loc='lower left')
leg2 = ax[1].legend(handles[:3], labels[:3], fontsize=10, ncol=2, loc='lower left')
leg3 = ax[2].legend(handles[3:-2], labels[3:-2], fontsize=10, ncol=2, loc='lower left')

for leg in [leg1, leg2, leg3]:
	leg.get_frame().set_edgecolor('k')
	leg.get_frame().set_linewidth(0.5)

plt.savefig('../figures/mass_fraction.pdf')