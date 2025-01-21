import glob
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

from dawn.sim_toolkit.profile_handler import HaloProfileHandler


def mass_integral(r, rho):
	return 4*np.pi*r**2*rho

files = glob.glob('../magneticum-data/data/profiles/Box2/hr_bao_cdm*nhalo2000.pkl')

profile_handler = HaloProfileHandler(['matter', 'cdm', 'gas', 'Pe', 'star'], files)





def get_binned_profiles(m_min, m_max, n_bins):
	mass_bin_edges = np.logspace(m_min, m_max, n_bins+1)
	r_grid = np.linspace(21, 1e3, 200)  # in kpc/h

	mean_profiles_matter, mean_profiles_cdm = [], []
	mean_profiles_gas, mean_profiles_star = [], []
	mean_profiles_Pe = []

	mean_rvirs, mean_mvirs  = [], []
	Nhalos = []

	for i in range(len(mass_bin_edges)-1):
		mmin, mmax = mass_bin_edges[i], mass_bin_edges[i+1]
		rmin = 0  # in kpc/h

		cdm_profs = profile_handler.get_masked_profile(mmin, mmax, rmin, 'cdm')
		gas_profs = profile_handler.get_masked_profile(mmin, mmax, rmin, 'gas')
		Pe_profs = profile_handler.get_masked_profile(mmin, mmax, rmin, 'Pe')
		star_profs = profile_handler.get_masked_profile(mmin, mmax, rmin, 'star')

		mean_rvir = np.average(cdm_profs.rvir, weights=cdm_profs.mvir)/1e3  # Convert to Mpc/h


		# Densities are in Msun/kpc^3 h^2
		Nhalos.append(len(cdm_profs.profile))
		print(f'Mmin = 10^{np.log10(mmin):.2f} Msun, Mmax = 10^{np.log10(mmax):.2f} Msun, Nhalos = {Nhalos[i]}')
		cdm_mean_prof = np.nanmean(cdm_profs.profile, axis=0)
		cdm_mean_r = np.nanmean(cdm_profs.rbins, axis=0)
		cdm_mean_x = np.nanmean(cdm_profs.xbins, axis=0)
		cdm_scatter = profile_handler.get_scatter(cdm_profs.profile, cdm_mean_prof)

		gas_mean_prof = np.nanmean(gas_profs.profile, axis=0)
		gas_mean_r = np.nanmean(gas_profs.rbins, axis=0)
		gas_mean_x = np.nanmean(gas_profs.xbins, axis=0)
		gas_scatter = profile_handler.get_scatter(gas_profs.profile, gas_mean_prof)

		Pe_mean_prof = np.nanmean(Pe_profs.profile, axis=0)
		Pe_mean_r = np.nanmean(Pe_profs.rbins, axis=0)
		Pe_mean_x = np.nanmean(Pe_profs.xbins, axis=0)
		Pe_scatter = profile_handler.get_scatter(Pe_profs.profile, Pe_mean_prof)

		star_mean_prof = np.nanmean(star_profs.profile, axis=0)
		star_mean_r = np.nanmean(star_profs.rbins, axis=0)
		star_mean_x = np.nanmean(star_profs.xbins, axis=0)
		star_scatter = profile_handler.get_scatter(star_profs.profile, star_mean_prof)

		matter_mean_prof = cdm_mean_prof + gas_mean_prof + star_mean_prof
		matter_mean_r = (cdm_mean_prof*cdm_mean_r +  gas_mean_prof* gas_mean_r + star_mean_prof*star_mean_r)/matter_mean_prof
		matter_mean_x = (cdm_mean_prof*cdm_mean_x +  gas_mean_prof* gas_mean_x + star_mean_prof*star_mean_x)/matter_mean_prof

		mean_profiles_matter.append([matter_mean_prof, matter_mean_r, matter_mean_x])
		mean_profiles_cdm.append([cdm_mean_prof, cdm_mean_r, cdm_mean_x, cdm_scatter])
		mean_profiles_gas.append([gas_mean_prof, gas_mean_r, gas_mean_x, gas_scatter])
		mean_profiles_Pe.append([Pe_mean_prof, Pe_mean_r, Pe_mean_x, Pe_scatter])
		mean_profiles_star.append([star_mean_prof, star_mean_r, star_mean_x, star_scatter])


		mask_nan = np.isnan(matter_mean_prof)
		xp, yp = matter_mean_r[~mask_nan], matter_mean_prof[~mask_nan]

		# Densities are in Msun/kpc^3 h^2
		prof_interpolator = interp1d(xp, yp, kind='cubic', fill_value=(0, 0), bounds_error=False)
		prof_interp = prof_interpolator(r_grid)

		mean_mvir = integrate.simpson(y=mass_integral(r_grid, prof_interp), x=r_grid)

		mean_rvirs.append(mean_rvir)
		mean_mvirs.append(mean_mvir)

	mean_profiles_cdm = np.array(mean_profiles_cdm)
	mean_profiles_gas = np.array(mean_profiles_gas)  
	mean_profiles_Pe = np.array(mean_profiles_Pe)*1e3 # in eV/cm^3h^2
	mean_profiles_star = np.array(mean_profiles_star)
	mean_profiles_matter = np.array(mean_profiles_matter)

	return mean_profiles_matter, mean_profiles_cdm, mean_profiles_gas, mean_profiles_Pe, mean_profiles_star, mean_rvirs, mean_mvirs, Nhalos