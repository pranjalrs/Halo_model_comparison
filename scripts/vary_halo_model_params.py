import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
import yaml

import pyccl as ccl

import mcmc
import multiprocessing as mp

from utils import get_param_grid

def plot_Pks(rescale=True):
	ncol = 1
	nrow = ndim
	fig = plt.figure(figsize=(18, ndim * 3.5))
	subfig = fig.subfigures(nrow, ncol, wspace=0.5)
	subfig = subfig.flatten()

	all_axes = []
	for i in range(len(subfig)):
		ax = subfig[i].subplots(1, len(fields))
		all_axes.append(ax)
	all_axes = np.atleast_2d(all_axes)

	denom_dict = {field: Pk_default_dict[field][0] if rescale else 1 for field in fields}

	for i, param in enumerate(fit_params):
		this_range = param_range_dict[param]

		normalize = mpl.colors.Normalize(vmin=min(this_range), vmax=max(this_range))
		scalar_map = plt.cm.ScalarMappable(norm=normalize, cmap=plt.cm.RdBu)

		for value_idx in range(ngrid):
			for j, field in enumerate(fields):
				all_axes[i, j].semilogx(k, Pk_dict[field][param][value_idx] / denom_dict[field],
										c=scalar_map.to_rgba(this_range[value_idx]))
				all_axes[i, j].text(0.1, 0.8, field, transform=all_axes[i, j].transAxes)

		fig.colorbar(scalar_map, ax=all_axes[i].ravel().tolist(), label=params_dict[param]['latex_name'])

	return all_axes

#---------------------------------------- 1. Load Config File ----------------------------------------#
run_parallel = False # Set this to False to run serially
halo_model = sys.argv[1]
config_path = f'../config/{halo_model}_all_default.yaml'

with open(config_path, 'r') as stream:
	config = yaml.safe_load(stream)

params_dict = config['params']
fit_params = list(params_dict.keys())
priors = [params_dict[param]['prior'] for param in fit_params]
latex_name = [params_dict[param]['latex_name'] for param in fit_params]

ndim = len(fit_params)

default_params_dict = {}  # bfg.Profiles.Mead20.Params_TAGN_7p8_MPr.copy()

fields = ['m-m', 'm-p', 'ne-ne', 'g-ne', 'xray', 'frb']
SAVE_PLOT = True
#---------------------------------------- 2. Initialize Model ----------------------------------------#
cosmo = ccl.CosmologyVanillaLCDM()
k = np.logspace(-2, 2, 500)
bfg_dict = mcmc.init_bfg_model(config['halo_model'], cosmo, a=1, k=k)
k = bfg_dict['k']
mcmc.bfg_dict = bfg_dict

Pk_default_dict = {field: mcmc.get_bfg_Pk(field=field, param_dict=default_params_dict) for field in fields}

print('P(k) for fiducial parameters computed...')

ngrid = 15
params_to_run = []

try:
	all_vary_dict = joblib.load(f'../data/{halo_model}/{halo_model}_all_default_vary_dict.pkl')
	Pk_dict = all_vary_dict.copy()
	param_range_dict = all_vary_dict['param_range_dict']

	fields_in_dict = list(Pk_dict.keys())
	params_in_dict = list(Pk_dict[fields[0]].keys())

	fields_to_run = [field for field in fields if field not in fields_in_dict]
	params_to_run = [param for param in fit_params if param not in params_in_dict]

	# Add all fields to run to Pk_dict
	for field in fields_to_run:
		Pk_dict[field] = {}

	if len(fields_to_run) != 0:
		params_to_run = fit_params

	print('[INFO]: Loading precomputed P(k)s for ...')
	print('[INFO]: Fields: ', fields_in_dict)
	print('[INFO]: Parameters: ', params_in_dict)

	if len(params_to_run) == 0 and len(fields_to_run) == 0:
		SAVE_PKL = False
	else:
		SAVE_PKL = True

except FileNotFoundError:
	print('Sampling grid...')
	Pk_dict = {field: {} for field in fields}
	Pk_dict['k'] = k
	Pk_dict['ngrid'] = ngrid
	param_range_dict = {}

	params_to_run = fit_params
	fields_to_run = fields

	all_vary_dict = {}

	SAVE_PKL = True

print('[INFO]: Parameters to run: ', params_to_run)
print('[INFO]: Fields to run: ', fields_to_run)
#---------------------------------------- 3. Compute P(k) ----------------------------------------#
def compute_Pk(param):
	param_range = get_param_grid(param, params_dict, ngrid)

	Pk_lists = {field: [] for field in fields}

	print('Computing P(k)s for ', param)
	for value in tqdm(param_range):
		for field in fields_to_run:
			Pk_lists[field].append(mcmc.get_bfg_Pk(value, param, field=field)[0])

	return param, Pk_lists, param_range

# Compute P(k) for each parameter
if run_parallel:
	with mp.Pool(processes=mp.cpu_count()) as pool:
		results = pool.map(compute_Pk, params_to_run)
else:
	results = [compute_Pk(param) for param in params_to_run]

# Unpack results and save in dictionary
for param, Pk_lists, param_range in results:
	for field in fields_to_run:
		Pk_dict[field][param] = Pk_lists[field]

		param_range_dict[param] = param_range

Pk_dict['param_range_dict'] = param_range_dict

if SAVE_PKL:
	joblib.dump(Pk_dict, f'../data/{halo_model}/{halo_model}_all_default_vary_dict.pkl')
#---------------------------------------- 3. Plot P(k) ----------------------------------------#
if SAVE_PLOT:  # Change to SAVE_PKL variable
	all_axes = plot_Pks(rescale=True)
	for i in range(ndim):
		all_axes[i, 0].set_ylabel('$P(k)/P(k)_\mathrm{fid}$')

	for ax in all_axes.flatten():
		ax.axhline(1, ls='--', c='k')
		ax.axhline(1.05, ls=':', c='gray')
		ax.axhline(0.95, ls=':', c='gray')
		ax.set_xlabel('$k [h/\mathrm{Mpc}]$')
		ax.set_ylim(0.48, 1.52)
	plt.savefig(f'../figures/{halo_model}/Pk/Param_vary_{halo_model}.pdf')
	plt.close()

	all_axes = plot_Pks(rescale=False)
	for i in range(ndim):
		all_axes[i, 0].set_ylabel('$P(k)$')
		ax.set_xlabel('$k [h/\mathrm{Mpc}]$')

	for ax in all_axes.flatten():
		ax.set_yscale('log')
	plt.savefig(f'../figures/{halo_model}/Pk/Param_vary_{halo_model}_no_rescale.pdf')
	plt.close()

#---------------------------------------- 4. Compute derivatives and plot----------------------------------------#
dlogPk_dlogtheta = {field: {} for field in fields}
for i in range(ndim):
	param = fit_params[i]
	param_range = param_range_dict[param]
	param_range = np.array(param_range)
	dx = (param_range[1] - param_range[0])

	if 'log' in param:
		mult_factor = 1
	else:
		mult_factor = param_range[:, np.newaxis]

	for field in fields:
		dlogPk_dlogtheta[field][param] = np.gradient(Pk_dict[field][param], dx, axis=0) / Pk_dict[field][param] * mult_factor

dlogPk_dlogtheta_data = {f'dlogPk_{field}_dlogtheta': dlogPk_dlogtheta[field] for field in fields}
dlogPk_dlogtheta_data.update({'k': k, 'params_range_dict': param_range_dict})

# Save
joblib.dump(dlogPk_dlogtheta_data, f'../data/{halo_model}/{halo_model}_dlogPk_dlogtheta_dict.pkl')

ncol = 1
nrow = ndim
fig = plt.figure(figsize=(15, ndim * 3.5))
subfig = fig.subfigures(nrow, ncol, wspace=0.5)
subfig = subfig.flatten()

all_axes = []
for i in range(len(subfig)):
	ax = subfig[i].subplots(1, len(fields))
	all_axes.append(ax)
all_axes = np.atleast_2d(all_axes)

normalize = mpl.colors.LogNorm(vmin=min(k), vmax=max(k))
scalar_map = plt.cm.ScalarMappable(norm=normalize, cmap=plt.cm.RdBu)

if SAVE_PLOT:
	print('[INFO]: Plotting dPk/dtheta...')
	for i in tqdm(range(ndim)):
		param = fit_params[i]
		this_range = param_range_dict[param]

		for value_idx in range(len(k)):
			for j, field in enumerate(fields):
				all_axes[i, j].plot(this_range, dlogPk_dlogtheta[field][param][:, value_idx],
									c=scalar_map.to_rgba(k[value_idx]))
				all_axes[i, j].text(0.1, 0.8, field, transform=all_axes[i, j].transAxes)

		this_latex_name = params_dict[param]['latex_name']
		for j in range(len(fields)):
			all_axes[i, j].set_xlabel(this_latex_name)

		fig.colorbar(scalar_map, ax=all_axes[i].ravel().tolist(), label='$k\, \, [h$/Mpc]')

		if 'log' not in this_latex_name:
			all_axes[i, 0].set_ylabel('$\\frac{\mathrm{d}\, \log P(k)}{\mathrm{d}\, \log ' + this_latex_name.strip('$') + '}$')
		else:
			all_axes[i, 0].set_ylabel('$\\frac{\mathrm{d}\, \log P(k)}{\mathrm{d}\, ' + this_latex_name.strip('$') + '}$')

plt.savefig(f'../figures/{halo_model}/Pk/{halo_model}_dPk_dtheta.pdf')
