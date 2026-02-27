import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from cosmo_funcs import (
    distance_modulus,
    absolute_magnitude_limit,
    get_all_comoving_volumes,
)
from luminosity_funcs import (
    simpson_integrate_with_params,
    robotham_11_func,
    luminosity_correction_factor,
    generate_hmf,
    ddm,
    bisection_ddm,
    get_zlims,
    histogram_numba,
    generate_empircal_lf,
    integrate_lf,
    cumulative_hmf,
    match_hmf_single,
    lf_to_hmf_match,
    update_halo_masses,
)

# Cosmology setup
cosmo_astropy = FlatLambdaCDM(H0=70, Om0=0.3)


def test_robotham_11_func_consistency():
    phi_star, M_star, alpha = 0.01, -20, -1
    M = -21
    val = robotham_11_func(M, phi_star, M_star, alpha)
    assert val > 0


def test_simpson_integrate_with_params_simple():
    phi_star, M_star, alpha = 0.01, -20, -1
    result = simpson_integrate_with_params(-25, -15, phi_star, M_star, alpha)
    assert result > 0


def test_luminosity_correction_factor_consistency():
    m_lim, z = 20, 0.1
    phi_star, M_star, alpha = 0.01, -20, -1
    omega_matter, h = 0.3, 0.7
    val = luminosity_correction_factor(
        m_lim, z, phi_star, M_star, alpha, omega_matter, h
    )
    assert val > 0


def test_generate_hmf_shapes():
    hmf_m, dn_dlogM = generate_hmf(0.1, 10, 15, 0.1, 0.7, 0.3)
    assert hmf_m.shape == dn_dlogM.shape
    assert np.all(dn_dlogM >= 0)


def test_ddm_bisection_and_get_zlims():
    abs_mag_val = -20
    k_corr_val = 0.0
    survey_mag_lim = 20
    omega_matter, h = 0.3, 0.7

    # Check ddm is decreasing with z
    z_test = 0.1
    val1 = ddm(z_test, abs_mag_val, k_corr_val, survey_mag_lim, omega_matter, h)
    val2 = ddm(z_test + 0.1, abs_mag_val, k_corr_val, survey_mag_lim, omega_matter, h)
    assert val1 > val2

    # Test bisection finds correct redshift
    z_lo, z_hi = 0.01, 1.0
    z_root = bisection_ddm(
        z_lo, z_hi, abs_mag_val, k_corr_val, survey_mag_lim, omega_matter, h
    )
    # The apparent magnitude at z_root should be close to survey limit
    m_app = abs_mag_val + distance_modulus(z_root, omega_matter, h) + k_corr_val
    np.testing.assert_allclose(m_app, survey_mag_lim, rtol=1e-3)

    # Test get_zlims returns reasonable array
    zs = np.array([0.01, 0.05])
    abs_mags = np.array([-19, -21])
    k_corrs = np.array([0.0, 0.0])
    z_max = 1.0
    zlims = get_zlims(zs, abs_mags, k_corrs, z_max, survey_mag_lim, omega_matter, h)
    assert zlims.shape == zs.shape
    for i in range(len(zs)):
        m_app_i = abs_mags[i] + distance_modulus(zlims[i], omega_matter, h) + k_corrs[i]
        assert m_app_i <= survey_mag_lim + 1e-3  # allow small tolerance


def test_histogram_numba_basic():
    x = np.array([0.1, 0.5, 1.2, 1.8])
    bins = np.array([0, 1, 2])
    counts = histogram_numba(x, bins)
    np.testing.assert_array_equal(counts, np.array([2, 2]))


def test_integrate_lf_basic():
    bins = np.array([-21, -20, -19])
    phi = np.array([1.0, 2.0])
    n = integrate_lf(phi, bins, -20.5)
    assert n > 0


def test_cumulative_hmf_basic():
    masses = np.array([1, 2, 3, 4])
    dn = np.array([1, 2, 3, 4])
    n_cum = cumulative_hmf(masses, dn)
    assert n_cum[0] > n_cum[-1]


def test_match_hmf_single_basic():
    masses = np.array([1, 2, 3, 4])
    dn = np.array([1, 2, 3, 4])
    n_target = 3.5
    M_thresh = match_hmf_single(n_target, masses, dn)
    assert M_thresh >= masses[0] and M_thresh <= masses[-1]


# Setup a simple cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


@pytest.fixture
def mock_galaxies_and_hmf():
    # Mock galaxy catalogue
    galaxies = {
        "abs_mag": np.array([-21, -20, -19, -18]),
        "z": np.array([0.05, 0.1, 0.2, 0.3]),
    }
    # Mock halo mass function
    hmf = {
        "logM": np.array([11, 12, 13, 14]),  # log10 halo mass
        "dn_dlogM": np.array([1e-3, 1e-4, 1e-5, 1e-6]),  # dn/dlogM
    }
    return galaxies, hmf


def test_lf_to_hmf_match_shapes_and_order():
    integral_mag_limits = np.array([-22, -21, -20])
    phi = np.array([0.1, 0.2, 0.3])
    bins = np.array([-23, -22, -21, -20])
    hmf_masses = np.array([1e12, 2e12, 3e12])
    dn_dlogM = np.array([0.01, 0.02, 0.03])

    halo_masses = lf_to_hmf_match(integral_mag_limits, phi, bins, hmf_masses, dn_dlogM)
    assert halo_masses.shape == integral_mag_limits.shape
    # check monotonicity
    assert np.all(np.diff(halo_masses) >= 0)


def test_lf_to_hmf_match_respects_z_limits():
    # use mags outside typical range to simulate "z limits" behavior
    integral_mag_limits = np.array([-25, -15])
    phi = np.array([0.1, 0.2, 0.3])
    bins = np.array([-23, -22, -21, -20])
    hmf_masses = np.array([1e12, 2e12, 3e12])
    dn_dlogM = np.array([0.01, 0.02, 0.03])

    halo_masses = lf_to_hmf_match(integral_mag_limits, phi, bins, hmf_masses, dn_dlogM)
    # Check that extreme magnitudes still return finite halo masses
    assert np.all(np.isfinite(halo_masses))


def test_update_halo_masses_runs_numpy():
    abs_mags = np.array([-22, -21])
    zs = np.array([0.1, 0.2])
    k_corrs = np.zeros(2)
    survey_mag_limit = -19
    survey_fractional_area = 0.5
    hmf_masses = np.array([1e12, 2e12])
    dn_dlogM = np.array([0.01, 0.02])
    omega_matter = 0.3
    h = 0.7

    masses = update_halo_masses(
        abs_mags,
        zs,
        k_corrs,
        survey_mag_limit,
        survey_fractional_area,
        hmf_masses,
        dn_dlogM,
        omega_matter,
        h,
    )

    assert masses.shape == abs_mags.shape
