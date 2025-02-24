import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from tqdm import tqdm
import yaml

import pyccl as ccl

import mcmc
import multiprocessing as mp

def get_param_grid(param, p_dict, ngrid):
	this_prior = p_dict[param]['prior']
	this_value = p_dict[param]['initial_value']

	dx = (this_prior[1] - this_prior[0])/ngrid

	assert ngrid%2!=0, 'ngrid must be odd'
	# Check if min and max possible values are within prior
	nstep_down = int((this_value - this_prior[0])//dx)
	nstep_up = int((this_prior[1] - this_value)//dx)

	if nstep_down + nstep_up + 1 > ngrid:
		if nstep_down > nstep_up:
			nstep_down -= 1
		else:
			nstep_up -= 1

	elif nstep_down + nstep_up + 1 < ngrid:
		if nstep_down < nstep_up:
			nstep_down += 1
		else:
			nstep_down += 1

	param_grid = [this_value - dx*i for i in range(nstep_down, 0, -1)]
	param_grid += [this_value]
	param_grid += [this_value + dx*i for i in range(1, nstep_up+1)]

	assert len(param_grid) == ngrid

	return param_grid

#---------------------------------------- 1. Load Config File ----------------------------------------#
run_parallel = False # Set this to False to run serially
halo_model = 'Mead20'
config_path = f'../config/{halo_model}_all_default.yaml'

with open(f'../config/{config_path}', 'r') as stream:
	config = yaml.safe_load(stream)

params_dict = config['params']
fit_params = list(params_dict.keys())
priors = [params_dict[param]['prior'] for param in fit_params]
latex_name = [params_dict[param]['latex_name'] for param in fit_params]

ndim = len(fit_params)

default_params_dict = {}  #bfg.Profiles.Mead20.Params_TAGN_7p8_MPr.copy()
#---------------------------------------- 2. Initialize Model ----------------------------------------#
cosmo = ccl.CosmologyVanillaLCDM()
bfg_dict = mcmc.init_bfg_model(config['halo_model'], cosmo, a=1)
k = bfg_dict['k']
mcmc.bfg_dict = bfg_dict

Pk_mm_default = mcmc.get_bfg_Pk(field='m-m', param_dict=default_params_dict)
Pk_mp_default = mcmc.get_bfg_Pk(field='m-p', param_dict=default_params_dict)
Pk_gas_default = mcmc.get_bfg_Pk(field='ne-ne', param_dict=default_params_dict)
Pk_xray_default = mcmc.get_bfg_Pk(field='xray', param_dict=default_params_dict)
Pk_frb_default = mcmc.get_bfg_Pk(field='frb', param_dict=default_params_dict)

print('P(k) for fiducial parameters computed...')

ngrid = 15
params_to_run = []

try:
	all_vary_dict = joblib.load(f'../data/{halo_model}/{halo_model}_all_default_vary_dict.pkl')
	Pk_mm_dict = all_vary_dict['Pk_mm_dict']
	Pk_mp_dict = all_vary_dict['Pk_mp_dict']
	Pk_gas_dict = all_vary_dict['Pk_gas_dict']
	Pk_xray_dict = all_vary_dict['Pk_xray_dict']
	Pk_frb_dict = all_vary_dict['Pk_frb_dict']
	param_range_dict = all_vary_dict['param_range_dict']
	params_to_run = [param for param in fit_params if param not in Pk_mm_dict.keys()]
	print('Loading precomputed P(k)s on grid...')

	if len(params_to_run) == 0:
		SAVE_PKL = False
	else:
		SAVE_PKL = True


except FileNotFoundError:
	print('Sampling grid...')
	Pk_mm_dict = {}
	Pk_mp_dict = {}
	Pk_gas_dict = {}
	Pk_xray_dict = {}
	Pk_frb_dict = {}
	param_range_dict = {}
	params_to_run = fit_params

	all_vary_dict = {}

	SAVE_PKL = True

#---------------------------------------- 3. Compute P(k) ----------------------------------------#
def compute_Pk(param):
	param_range = get_param_grid(param, params_dict, ngrid)
	param_range_dict[param] = param_range

	Pk_mm_list = []
	Pk_mp_list = []
	Pk_gas_list = []
	Pk_xray_list = []
	Pk_frb_list = []

	print('Computing P(k)s for ', param)
	for value in tqdm(param_range):
		Pk_mm_list.append(mcmc.get_bfg_Pk(value, param, field='m-m')[0])
		Pk_mp_list.append(mcmc.get_bfg_Pk(value, param, field='m-p')[0])
		Pk_gas_list.append(mcmc.get_bfg_Pk(value, param, field='ne-ne')[0])
		Pk_xray_list.append(mcmc.get_bfg_Pk(value, param, field='xray')[0])
		Pk_frb_list.append(mcmc.get_bfg_Pk(value, param, field='frb')[0])

	return param, Pk_mm_list, Pk_mp_list, Pk_gas_list, Pk_xray_list, Pk_frb_list, param_range


if run_parallel:
	with mp.Pool(processes=mp.cpu_count()) as pool:
		results = pool.map(compute_Pk, params_to_run)
else:
	results = [compute_Pk(param) for param in params_to_run]

for param, Pk_mm_list, Pk_mp_list, Pk_gas_list, Pk_xray_list, Pk_frb_list, param_range in results:
	Pk_mm_dict[param] = Pk_mm_list
	Pk_mp_dict[param] = Pk_mp_list
	Pk_gas_dict[param] = Pk_gas_list
	Pk_xray_dict[param] = Pk_xray_list
	Pk_frb_dict[param] = Pk_frb_list
	param_range_dict[param] = param_range

all_vary_dict['Pk_mm_dict'] = Pk_mm_dict
all_vary_dict['Pk_mp_dict'] = Pk_mp_dict
all_vary_dict['Pk_gas_dict'] = Pk_gas_dict
all_vary_dict['Pk_xray_dict'] = Pk_xray_dict
all_vary_dict['Pk_frb_dict'] = Pk_frb_dict
all_vary_dict['param_range_dict'] = param_range_dict

joblib.dump(all_vary_dict, f'../data/{halo_model}/{halo_model}_all_default_vary_dict.pkl')

# if SAVE_PKL:
# 	joblib.dump(all_vary_dict, f'../data/{halo_model}/{halo_model}_all_default_vary_dict.pkl')

#---------------------------------------- 3. Plot P(k) ----------------------------------------#
if SAVE_PKL:
	ncol = 1
	nrow = ndim
	fig = plt.figure(figsize=(18, ndim*3.5))
	subfig = fig.subfigures(nrow, ncol, wspace=0.5)
	subfig = subfig.flatten()

	all_axes = []
	for i in range(len(subfig)):
		ax = subfig[i].subplots(1, 5)

		all_axes.append(ax)
	all_axes = np.array(all_axes)

	for i, param in enumerate(fit_params):
		this_range = param_range_dict[param]

		normalize = mpl.colors.Normalize(vmin=min(this_range), vmax=max(this_range))
		scalar_map = plt.cm.ScalarMappable(norm=normalize, cmap=plt.cm.RdBu)


		for value_idx in range(ngrid):
			all_axes[i, 0].semilogx(k, Pk_mm_dict[param][value_idx]/Pk_mm_default[0],
									c=scalar_map.to_rgba(this_range[value_idx]))
			all_axes[i, 1].semilogx(k, Pk_mp_dict[param][value_idx]/Pk_mp_default[0],
									c=scalar_map.to_rgba(this_range[value_idx]))
			all_axes[i, 2].semilogx(k, Pk_gas_dict[param][value_idx]/Pk_gas_default[0],
									c=scalar_map.to_rgba(this_range[value_idx]))
			all_axes[i, 3].semilogx(k, Pk_xray_dict[param][value_idx]/Pk_xray_default[0],
									c=scalar_map.to_rgba(this_range[value_idx]))
			all_axes[i, 4].semilogx(k, Pk_frb_dict[param][value_idx]/Pk_frb_default[0],
									c=scalar_map.to_rgba(this_range[value_idx]))

			all_axes[i, 0].text(0.1, 0.8, 'm-m\n (cosmic shear)', transform=all_axes[i, 0].transAxes)
			all_axes[i, 1].text(0.1, 0.8, 'm-y\n (shear x tSZ)', transform=all_axes[i, 1].transAxes)
			all_axes[i, 2].text(0.1, 0.8, '$n_\mathrm{e}-n_\mathrm{e}$\n (kSZ)', transform=all_axes[i, 2].transAxes)
			all_axes[i, 3].text(0.1, 0.8, 'X-ray', transform=all_axes[i, 3].transAxes)
			all_axes[i, 4].text(0.1, 0.8, 'FRB', transform=all_axes[i, 4].transAxes)

	#     fig.subplots_adjust(right=0.8)
	#     cbar_ax = subfig[i].add_axes([0.2, 0.9, 0.7, 0.05])
	#     fig.colorbar(scalar_map, orientation='horizontal', cax=cbar_ax, location='top', label=param_latex_dict[name])
		fig.colorbar(scalar_map, ax=all_axes[i].ravel().tolist(), label=params_dict[param]['latex_name'])

		all_axes[i, 0].set_ylabel('$P(k)/P(k)_\mathrm{fid}$')

	for ax in all_axes.flatten():
		ax.axhline(1, ls='--', c='k')
		ax.axhline(1.05, ls=':', c='gray')
		ax.axhline(0.95, ls=':', c='gray')
		ax.set_xlabel('$k [h/\mathrm{Mpc}]$')
		ax.set_ylim(0.48, 1.52)

	plt.savefig(f'../figures/{halo_model}/Param_vary_{halo_model}.pdf')
	plt.close()

#---------------------------------------- 4. Compute derivatives and plot----------------------------------------#
dlogPkmm_dlogtheta = {}
dlogPkmp_dlogtheta = {}
dlogPkgas_dlogtheta = {}
dlogPkxray_dlogtheta = {}
dlogPkfrb_dlogtheta = {}
for i in range(ndim):
	param = fit_params[i]
	param_range = param_range_dict[param]
	param_range = np.array(param_range)
	dx = (param_range[1] - param_range[0])

	if 'log' in param: mult_factor = 1
	else: mult_factor = param_range[:, np.newaxis]

	dlogPkmm_dlogtheta[param] = np.gradient(Pk_mm_dict[param], dx, axis=0)/Pk_mm_dict[param] * mult_factor
	dlogPkmp_dlogtheta[param] = np.gradient(Pk_mp_dict[param], dx, axis=0)/Pk_mp_dict[param] * mult_factor
	dlogPkgas_dlogtheta[param] = np.gradient(Pk_gas_dict[param], dx, axis=0)/Pk_gas_dict[param] * mult_factor
	dlogPkxray_dlogtheta[param] = np.gradient(Pk_xray_dict[param], dx, axis=0)/Pk_xray_dict[param] * mult_factor
	dlogPkfrb_dlogtheta[param] = np.gradient(Pk_frb_dict[param], dx, axis=0)/Pk_frb_dict[param] * mult_factor


dlogPk_dlogtheta_data = {'dlogPk_mm_dlogtheta': dlogPkmm_dlogtheta,
						 'dlogPk_mp_dlogtheta': dlogPkmp_dlogtheta,
						 'dlogPk_gas_dlogtheta': dlogPkgas_dlogtheta,
						 'dlogPk_xray_dlogtheta': dlogPkxray_dlogtheta,
						 'dlogPk_frb_dlogtheta': dlogPkfrb_dlogtheta,
						 'k': k,
						 'params_range_dict': param_range_dict}

# Save
joblib.dump(dlogPk_dlogtheta_data, f'../data/{halo_model}/{halo_model}_dlogPk_dlogtheta_dict.pkl')

# fig, ax = plt.subplots(nparam, 1, figsize=(12, nparam*3))
ncol = 1
nrow = ndim
fig = plt.figure(figsize=(15, ndim*3.5))
subfig = fig.subfigures(nrow, ncol, wspace=0.5)
subfig = subfig.flatten()

all_axes = []
for i in range(len(subfig)):
	ax = subfig[i].subplots(1, 5)

	all_axes.append(ax)
all_axes = np.array(all_axes)

normalize = mpl.colors.LogNorm(vmin=min(k), vmax=max(k))
scalar_map = plt.cm.ScalarMappable(norm=normalize, cmap=plt.cm.RdBu)


if True:
	for i in range(ndim):
		param = fit_params[i]
		this_range = param_range_dict[param]


		for value_idx in range(len(k)):
			all_axes[i, 0].plot(this_range, dlogPkmm_dlogtheta[param][:, value_idx],
									c=scalar_map.to_rgba(k[value_idx]))
			all_axes[i, 1].plot(this_range, dlogPkmp_dlogtheta[param][:, value_idx],
									c=scalar_map.to_rgba(k[value_idx]))
			all_axes[i, 2].plot(this_range, dlogPkgas_dlogtheta[param][:, value_idx],
									c=scalar_map.to_rgba(k[value_idx]))
			all_axes[i, 3].plot(this_range, dlogPkxray_dlogtheta[param][:, value_idx],
									c=scalar_map.to_rgba(k[value_idx]))
			all_axes[i, 4].plot(this_range, dlogPkfrb_dlogtheta[param][:, value_idx],
									c=scalar_map.to_rgba(k[value_idx]))

			all_axes[i, 0].text(0.1, 0.8, 'm-m', transform=all_axes[i, 0].transAxes)
			all_axes[i, 1].text(0.1, 0.8, 'm-y', transform=all_axes[i, 1].transAxes)
			all_axes[i, 2].text(0.1, 0.8, '$n_\mathrm{e}-n_\mathrm{e}$', transform=all_axes[i, 2].transAxes)
			all_axes[i, 3].text(0.1, 0.8, 'X-ray', transform=all_axes[i, 3].transAxes)
			all_axes[i, 4].text(0.1, 0.8, 'FRB', transform=all_axes[i, 4].transAxes)

			this_latex_name = params_dict[param]['latex_name']
			all_axes[i, 0].set_xlabel(this_latex_name)
			all_axes[i, 1].set_xlabel(this_latex_name)
			all_axes[i, 2].set_xlabel(this_latex_name)
			all_axes[i, 3].set_xlabel(this_latex_name)
			all_axes[i, 4].set_xlabel(this_latex_name)

	#     fig.subplots_adjust(right=0.8)
	#     cbar_ax = subfig[i].add_axes([0.2, 0.9, 0.7, 0.05])
	#     fig.colorbar(scalar_map, orientation='horizontal', cax=cbar_ax, location='top', label=param_latex_dict[name])
		fig.colorbar(scalar_map, ax=all_axes[i].ravel().tolist(), label='$k\, \, [h$/Mpc]')

		if 'log' not in this_latex_name:
			all_axes[i, 0].set_ylabel('$\\frac{\mathrm{d}\, \log P(k)}{\mathrm{d}\, \log '+this_latex_name.strip('$')+'}$')

		else:
			all_axes[i, 0].set_ylabel('$\\frac{\mathrm{d}\, \log P(k)}{\mathrm{d}\, '+this_latex_name.strip('$')+'}$')

	plt.savefig(f'../figures/{halo_model}/{halo_model}_dPk_dtheta.pdf')
