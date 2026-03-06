import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from hmf import MassFunction
from hmf import cosmo
from astropy.cosmology import FlatLambdaCDM
import logging
from numba import njit, prange
from cosmo_funcs import (
    distance_modulus,
    get_all_comoving_volumes,
    absolute_magnitude_limit,
)


@njit
def simpson_integrate_with_params(a, b, phi_star, M_star, alpha, n=1000):
    """Simpson's rule integration with parameters passed directly"""
    if n % 2 == 1:
        n += 1  # Ensure n is even
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    for i in range(n + 1):
        y[i] = robotham_11_func(x[i], phi_star, M_star, alpha)
    result = y[0] + y[n]
    for i in range(1, n, 2):
        result += 4 * y[i]
    for i in range(2, n, 2):
        result += 2 * y[i]
    return result * h / 3


@njit
def robotham_11_func(M, phi_star, M_star, alpha):
    """Fixed numba function with proper parameters"""
    return (
        10 ** (-0.4 * M)
        * 0.4
        * np.log(10)
        * phi_star
        * (10 ** (0.4 * (M_star - M))) ** (alpha + 1)
        * np.exp(-(10 ** (0.4 * (M_star - M))))
    )


@njit
def luminosity_correction_factor(m_lim, z, phi_star, M_star, alpha, omega_matter, h):
    """Numba-compiled function using custom Simpson's rule integration
    Parameters:
        m_lim (float): Apparent magnitude limit of the survey
        z (float): Redshift at which to compute the correction
        phi_star (float): Schechter function parameter
        M_star (float): Schechter function parameter
        alpha (float): Schechter function parameter
        omega_matter (float): Matter density parameter at z=0
        h (float): Dimensionless Hubble parameter
    Returns:
        float: Luminosity correction factor"""

    lf_M_lim = absolute_magnitude_limit(z, m_lim, omega_matter, h)

    # Use Simpson's rule integration with parameters passed directly
    int_lf_total = simpson_integrate_with_params(-30.0, -14.0, phi_star, M_star, alpha)
    int_lf_to_lim = simpson_integrate_with_params(
        -30.0, lf_M_lim, phi_star, M_star, alpha
    )

    return 1.04 * int_lf_total / int_lf_to_lim


def generate_hmf(hmf_z, m_min, m_max, dlog10m, h, omega_matter):
    """Generate the halo mass function at a given redshift.
    Parameters:
        hmf_z (float): Redshift at which to compute the HMF
        m_min (float): Minimum halo mass in log10(Msun/h)
        m_max (float): Maximum halo mass in log10(Msun/h)
        dlog10m (float): Mass bin width in dex
        h (float): Dimensionless Hubble parameter
        omega_matter (float): Matter density parameter
    Returns:
        tuple: (halo_masses, dn_dlogM) where halo_masses is an array of
            log10 halo masses and dn_dlogM is the differential number density
    """

    astropy_cosmo = FlatLambdaCDM(H0=h * 100, Om0=omega_matter)
    hmf_cosmo = cosmo.Cosmology(cosmo_model=astropy_cosmo)
    mf = MassFunction(z=hmf_z, Mmin=m_min, Mmax=m_max, dlog10m=dlog10m)
    return mf.m, mf.dndlog10m


@njit
def ddm(z, abs_mag_val, k_corr, survey_mag_lim, omega_matter, h):
    """
    Difference between apparent magnitude limit and absolute magnitude at redshift z.
    Parameters
    ----------
    z : float
        Redshift.
    abs_mag_val : float
        Absolute magnitude of the galaxy.
    k_corr_val : float
        K-correction value.
    survey_mag_lim : float
        Apparent magnitude limit of the survey.
    omega_matter : float
        Matter density parameter.
    h : float
        Dimensionless Hubble parameter.
    Returns
    -------
    float
        The difference between the survey magnitude limit and the computed apparent magnitude.

    """
    return survey_mag_lim - abs_mag_val - distance_modulus(z, omega_matter, h) - k_corr


@njit
def bisection_ddm(
    z_lo,
    z_hi,
    abs_mag_val,
    k_corr_val,
    survey_mag_lim,
    omega_matter,
    h,
    tol=1e-5,
    maxiter=100,
):
    """
    Simple bisection method to solve ddm(z) = 0 between [z_lo, z_hi].
    Parameters
    ----------
    z_lo : float
        Lower bound of redshift.
    z_hi : float
        Upper bound of redshift.
    mag_lim : float
        Apparent magnitude limit of the survey.
    abs_mag_val : float
        Absolute magnitude of the galaxy.
    k_corr_val : float
        K-correction value.
    survey_mag_lim : float
        Apparent magnitude limit of the survey.
    omega_matter : float
        Matter density parameter.
    h : float
        Dimensionless Hubble parameter.
    tol : float
        Tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    Returns
    -------
    float
        Redshift where ddm(z) = 0, or np.nan if no root is found in the interval.
    """
    f_lo = ddm(z_lo, abs_mag_val, k_corr_val, survey_mag_lim, omega_matter, h)
    f_hi = ddm(z_hi, abs_mag_val, k_corr_val, survey_mag_lim, omega_matter, h)

    if f_lo * f_hi > 0:
        return np.nan  # no root in [z_lo, z_hi]

    for _ in range(maxiter):
        mid = 0.5 * (z_lo + z_hi)
        f_mid = ddm(mid, abs_mag_val, k_corr_val, survey_mag_lim, omega_matter, h)
        if abs(f_mid) < tol or (z_hi - z_lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            z_hi = mid
            f_hi = f_mid
        else:
            z_lo = mid
            f_lo = f_mid
    return 0.5 * (z_lo + z_hi)


@njit
def get_zlims(zs, abs_mags, k_corrs, z_max, survey_mag_lim, omega_matter, h):
    """
    Compute redshift limits for each galaxy.

    Parameters
    ----------
    zs : array
        Current redshifts of galaxies (lower bound for root finding).
    abs_mags : array
        Absolute magnitudes of galaxies.
    k_corrs : array
        K-corrections for galaxies.
    z_max : float
        Maximum redshift of the survey.
    survey_mag_lim : float
        Apparent magnitude limit of the survey.
    omega_matter : float
        Matter density parameter.
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    zlim : array
        Observable redshift limits for each galaxy.
    """
    n = len(zs)
    zlim = np.zeros(n)

    for i in range(n):
        abs_val = abs_mags[i]
        k_val = k_corrs[i]

        # If still visible at z_max, vmax uses z_max
        if ddm(z_max, abs_val, k_val, survey_mag_lim, omega_matter, h) > 0.0:
            zlim[i] = z_max
        else:
            # Root should lie in [0, z_max] (if it exists)
            z_root = bisection_ddm(
                0.0, z_max, abs_val, k_val, survey_mag_lim, omega_matter, h
            )
            if np.isnan(z_root):
                # If no root, safest is zlim=zs (gives minimal vmax, avoids crushing phi)
                zlim[i] = zs[i]
            else:
                zlim[i] = z_root

    return zlim


@njit
def histogram_numba(x, bins, weights=None):
    """
    A simple implementation of histogram using Numba for speedup.
    Parameters
    ----------
    x : array
        Input data to be binned.
    bins : array
        Bin edges.
    weights : array, optional
        Weights for each data point.
    Returns
    -------
    counts : array
        Counts in each bin.
    """
    n_bins = len(bins) - 1
    counts = np.zeros(n_bins)

    for i in range(x.size):
        val = x[i]
        w = 1.0 if weights is None else weights[i]

        # skip values outside bin range (like np.histogram does)
        if val < bins[0] or val >= bins[-1]:
            continue

        # find bin index using linear search
        for j in range(n_bins):
            if bins[j] <= val < bins[j + 1]:
                counts[j] += w
                break

    return counts


@njit
def generate_empircal_lf(
    group_abs_mags,
    group_zs,
    bcg_abs_mags,
    bcg_k_corrs,
    survey_mag_lim,
    survey_area_fraction,
    omega_matter,
    h,
):
    """Generate empirical luminosity function using 1/Vmax method.
    Parameters
    ----------
    group_abs_mags : array
        Absolute magnitude of the groups.
    group_zs : array
        Redshifts of galaxies.
    bcg_abs_mags : array
        Absolute magnitudes of brightest cluster galaxies.
    bcg_k_corrs : array
        K-corrections for galaxies.
    survey_mag_lim : float
        Apparent magnitude limit of the survey.
    survey_area_fraction : float
        Fraction of the sky covered by the survey.
    omega_matter : float
        Matter density parameter.
    h : float
        Dimensionless Hubble parameter.
    Returns
    -------
    phi : array
        Luminosity function values in each magnitude bin.
    bins : array
        Magnitude bins.
    """
    abs_mag_max = np.max(group_abs_mags)
    abs_mag_min = np.min(group_abs_mags)
    z_max = np.max(group_zs)

    zlims = get_zlims(
        group_zs, bcg_abs_mags, bcg_k_corrs, z_max, survey_mag_lim, omega_matter, h
    )  # zs, abs_mags, k_corrs, z_max, survey_mag_lim

    # vs = survey_area_fraction * get_all_comoving_volumes(zs, omega_matter)

    vmaxs = survey_area_fraction * get_all_comoving_volumes(zlims, omega_matter)

    # v_vmaxs = vs / vmaxs

    bins = np.linspace(abs_mag_min, abs_mag_max, 50)

    phi = histogram_numba(group_abs_mags, bins=bins, weights=1.0 / vmaxs)
    return phi, bins


@njit
def integrate_lf(phi, bins, integral_mag_limit):
    """
    Integrate luminosity function down to mag_limit.
    Returns number density in h^3 Mpc^-3.
    Parameters
    ----------
    phi : array
        Luminosity function values in each magnitude bin.
    bins : array
        Magnitude bins.
    integral_mag_limit : float
        Absolute magnitude limit for integration.
    Returns
    -------
    float
        Integrated number density down to mag_limit in h^3 Mpc^-3.
    """
    # bins are ascending: bright (more negative) -> faint
    if integral_mag_limit <= bins[0]:
        return 0.0
    if integral_mag_limit >= bins[-1]:
        # integrate full range
        bin_widths = np.diff(bins)
        return np.sum(phi * bin_widths)

    bin_widths = np.diff(bins)

    idx = np.searchsorted(bins, integral_mag_limit) - 1
    if idx < 0:
        idx = 0
    if idx >= phi.size:
        idx = phi.size - 1

    # full bins strictly brighter than limit
    full_contrib = np.sum(phi[:idx] * bin_widths[:idx])

    # partial bin
    m_left = bins[idx]
    m_right = bins[idx + 1]
    frac = (integral_mag_limit - m_left) / (m_right - m_left)

    # constant within bin (histogram)
    partial_contrib = phi[idx] * (integral_mag_limit - m_left)

    return full_contrib + partial_contrib


@njit
def cumulative_hmf(hmf_masses, dn_dlogM):
    """
    Compute cumulative HMF n(>M) with trapezoidal rule.
    Returns array same length as M.
    Parameters
    ----------
    hmf_masses : array
        Halo masses in h^-1 Msun.
    dn_dlogM : array
        Differential HMF dn/dlogM in h^3 Mpc^-3.
    Returns
    -------
    n_cum : array
        Cumulative number density n(>M) in h^3 Mpc^-3.
    """
    logM = np.log10(hmf_masses)
    dlogM = np.diff(logM)
    trap = 0.5 * (dn_dlogM[1:] + dn_dlogM[:-1]) * dlogM
    n_cum = np.zeros_like(hmf_masses)
    n_cum[:-1] = np.flip(np.cumsum(np.flip(trap)))
    return n_cum


@njit
def match_hmf_single(n_target, hmf_masses, dn_dlogM):
    """
    Find M threshold corresponding to n_target via linear interpolation.
    Parameters
    ----------
    n_target : float
        Target number density in h^3 Mpc^-3.
    hmf_masses : array
        Halo masses in h^-1 Msun.
    dn_dlogM : array
        Differential HMF dn/dlogM in h^3 Mpc^-3.
    Returns
    -------
    M_thresh : float
        Halo mass threshold in h^-1 Msun.
    """
    n_cum = cumulative_hmf(hmf_masses, dn_dlogM)
    logM = np.log10(hmf_masses)

    # If outside range, clamp instead of always max-mass
    n_hi = n_cum[0]  # at low mass
    n_lo = n_cum[-2]  # last non-zero-ish (since last is 0)
    if n_target >= n_hi:
        return hmf_masses[0]  # very abundant -> low mass
    if n_target <= n_lo:
        return hmf_masses[-1]  # very rare -> high mass

    for i in range(len(hmf_masses) - 1):
        n1, n2 = n_cum[i], n_cum[i + 1]
        if (n1 >= n_target >= n2) or (n1 <= n_target <= n2):
            frac = (n_target - n1) / (n2 - n1)
            logM_thresh = logM[i] + frac * (logM[i + 1] - logM[i])
            return 10.0**logM_thresh

    return hmf_masses[-1]


@njit(parallel=True)
def lf_to_hmf_match(group_integral_mag_limits, phi, bins, hmf_masses, dn_dlogM):
    """
    Wrapper: for array of mag_limits, return array of halo mass thresholds.
    Parameters
    ----------
    group_integral_mag_limits : array
        Array of absolute magnitude limits for integration.

    phi : array
        Luminosity function values in each magnitude bin.
    bins : array
        Magnitude bins.
    hmf_masses : array
        Halo masses in h^-1 Msun.
    dn_dlogM : array
        Differential HMF dn/dlogM in h^3 Mpc^-3.


    Returns
    -------
    halo_masses : array

        Array of halo mass thresholds in h^-1 Msun corresponding to each integral_mag_limit.
    """

    halo_masses = np.empty(len(group_integral_mag_limits))

    for i in prange(len(group_integral_mag_limits)):
        halo_masses[i] = match_hmf_single(
            integrate_lf(phi, bins, group_integral_mag_limits[i]), hmf_masses, dn_dlogM
        )

    return halo_masses


@njit
def abundance_match_halo_masses(
    abs_mags,
    zs,
    bcg_abs_mags,
    bcg_k_corrs,
    survey_mag_limit,
    survey_fractional_area,
    hmf_masses,
    dn_dlogM,
    omega_matter,
    h,
):
    """
    Main function to update halo masses based on luminosity function matching.
    Parameters
    ----------
    abs_mags : array
        Absolute magnitudes of galaxies.
    zs : array
        Redshifts of galaxies.
    bcg_abs_mags : array
        Absolute magnitudes of brightest cluster galaxies.
    bcg_k_corrs : array
        K-corrections for galaxies.
    survey_mag_limit : float
        Apparent magnitude limit of the survey.
    survey_fractional_area : float
        Fraction of the sky covered by the survey.
    hmf_masses : array
        Halo masses in h^-1 Msun.
    dn_dlogM : array
        Differential HMF dn/dlogM in h^3 Mpc^-3.
    omega_matter : float
        Matter density parameter.
    h : float
        Dimensionless Hubble parameter.
    Returns
    -------
    matched_masses : array
        Array of halo mass thresholds in log10(Msun/h) corresponding to each galaxy.
    """
    phi, bins = generate_empircal_lf(
        abs_mags,
        zs,
        bcg_abs_mags,
        bcg_k_corrs,
        survey_mag_limit,
        survey_fractional_area,
        omega_matter,
        h,
    )

    matched_masses = lf_to_hmf_match(abs_mags, phi, bins, hmf_masses, dn_dlogM)

    return np.log10(matched_masses)


@njit
def k_corr(zs):
    # K-correction from Robotham+11
    z_ref = 0
    Q_z_ref = 1.75
    z_p = 0.2
    N = 4
    a = [0.2085, 1.0226, 0.5237, 3.5902, 2.3843]

    k_corrs = np.zeros(len(zs))
    for j in range(len(zs)):
        zspec = zs[j]
        k_e = Q_z_ref * (zspec - z_ref)
        for i in range(N + 1):
            k_e += a[i] * ((zspec - z_p) ** i)
        k_corrs[j] = k_e
    return k_corrs

@njit
def linear_stellar_mass2halo_mass(stellar_masses, intercept, slope):
    """Simple linear relation between stellar mass and halo mass.
    Parameters:
    stellar_masses : array
        Stellar masses of galaxies in units of h^-1 Msun.
    intercept : float
        Intercept of the linear relation in log-log space.
    slope : float
        Slope of the linear relation in log-log space.
    Returns:
    halo_masses : array
        Estimated halo masses in units of log10(Msun/h).
    """
    halo_masses = np.empty_like(stellar_masses)
    for i in prange(len(stellar_masses)):
        halo_masses[i] = intercept + slope * np.log10(stellar_masses[i])
    return halo_masses


@njit
def linear_luminosity2halo_mass(luminosities, intercept, slope):
    """Simple linear relation between luminosity and halo mass.
    Parameters:
    luminosities : array
        Luminosities of galaxies in units of h^-1 L_sun.
        intercept : float
        Intercept of the linear relation in log-log space.
        slope : float
        Slope of the linear relation in log-log space.
        Returns:
        halo_masses : array
        Estimated halo masses in units of log10(Msun/h).
    """
    halo_masses = np.empty_like(luminosities)
    for i in prange(len(luminosities)):
        halo_masses[i] = intercept + slope * np.log10(luminosities[i] * 1e14)
    return halo_masses


@njit
def red_blue_linear_luminosity2halo_mass(luminosities, central_is_red, intercept_red, slope_red, intercept_blue, slope_blue):
    """Separate linear relations for red and blue centrals.
    Parameters:
    luminosities : array
        Luminosities of galaxies in units of h^-1 L_sun.
        central_is_red : array
        Boolean array indicating if each galaxy is a red central.
        intercept_red : float
        Intercept for red centrals in log-log space.
        slope_red : float
        Slope for red centrals in log-log space.
        intercept_blue : float
        Intercept for blue centrals in log-log space.
        slope_blue : float
        Slope for blue centrals in log-log space.
        Returns:
        halo_masses : array
        Estimated halo masses in units of log10(Msun/h).
    """
    halo_masses = np.empty_like(luminosities)
    for i in prange(len(luminosities)):
        if central_is_red[i]:
            halo_masses[i] = intercept_red + slope_red * np.log10(luminosities[i] * 1e14)
        else:
            halo_masses[i] = intercept_blue + slope_blue * np.log10(luminosities[i] * 1e14)
    return halo_masses


@njit
def stellar2halo_mass_van_kampen(group_stellar_mass_3_largest, A = 46.944, logM_A = 10.483, beta = 0.249, gamma = -0.601):
    """Van Kampen+2026 relation between stellar mass and halo mass.
    Parameters:
    group_stellar_mass_3_largest : array
        Stellar mass of the 3 largest galaxies in each group in units of h^-1 Msun.
    A : float
        Normalization constant (default 46.944).
    logM_A : float
        Logarithm of the characteristic mass M_A in units of log10(Msun/h) (default 10.483).
    beta : float
        Slope of the low-mass end (default 0.249).
    gamma : float
        Slope of the high-mass end (default -0.601).
    Returns:
    halo_masses : array
        Estimated halo masses in units of log10(Msun/h).
    """
    M_A = 10.0 ** logM_A
    return np.log10(A * group_stellar_mass_3_largest * ((group_stellar_mass_3_largest/M_A)**beta+(group_stellar_mass_3_largest/M_A)**gamma))
