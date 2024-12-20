import numpy as np

import pyccl as ccl
import pyhmcode


def log_likelihood(k, Pk_data):


	hmcode_pofk = pyhmcode.calculate_nonlinear_power_spectrum(
									cosmology=hmcode_cosmology,
									halomodel=hmcode_model,
									fields=[pyhmcode.field_matter,
											pyhmcode.field_electron_pressure])

	# The output of calculate_nonlinear_power_spectrum has
	# shape (n_field, n_field, n_z, n_k).
	matter_matter_pofk = hmcode_pofk[0, 0]
	# matter_electron_pressure_pofk = hmcode_pofk[0, 1]

	matter_matter_pofk = np.interp(k, k, matter_matter_pofk)
	# Compute the chi^2
	chi2 = np.sum((hmcode_pofk - Pk_data)**2/variance)

	return -0.5*chi2


#----------------------------------- 1. Setup Cosmology and Halo Model -----------------------------------#
ccl_cosmology = ccl.CosmologyVanillaLCDM()

k = np.logspace(-4, 1.5, 100)
a = np.linspace(1/(1+6), 1, 10)
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

# Create the halo model object, which holds information on the specific halo
# model to use. E.g., the HMCode or HMx version.
hmcode_model = pyhmcode.Halomodel(
					pyhmcode.HMx2020_matter_pressure_w_temp_scaling)

#----------------------------------- 2. Load Data -----------------------------------#
Pk_magneticum, box_size = np.loadtxt('../magneticum-data/data/Pylians/Pk_matter/Box2/Pk_hr_bao_CIC_R2048.txt'), 896
# Pk_magneticum, box_size = np.loadtxt('../../magneticum-data/data/Pylians/Pk_matter/Box2/Pk_hr_bao_CIC_R1024.txt'), 352

k = Pk_magneticum[:, 0]
Pk_sim = Pk_magneticum[:, 1]
kmax = 6 # h/Mpc
Pk_sim,	k = Pk_sim[k < kmax], k[k < kmax]

delta_k = 2*np.pi/box_size
Nk = 2*np.pi * (k/delta_k)**2
variance = Pk_sim**2/Nk