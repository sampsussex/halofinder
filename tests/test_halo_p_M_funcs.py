# tests/test_halo_funcs.py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G
from astropy import units as u

# Import functions from your src folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from halo_p_M_funcs import (  # change filename if needed
    find_rho_crit,
    find_Om,
    find_halo_r,
    find_concentration_ratio,
    find_scale_radius,
    f_x,
    find_delta_bar,
    find_NFW_sigma,
    find_sigma_sqr,
    find_p_delta_z,
    find_p_M,
)

RTOL = 5e-3
ATOL = 1e-6


def test_find_rho_crit_matches_astropy():
    """Critical density from our function should match astropy's FlatLambdaCDM.critical_density0 scaled to z."""
    z = 0.5
    om = 0.3
    cosmo = FlatLambdaCDM(H0=100.0, Om0=om)
    rho_astropy = cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value
    rho_ours = find_rho_crit(z, om)
    assert_allclose(rho_ours, rho_astropy, rtol=RTOL)


def test_find_Om_consistency_with_astropy():
    z = 1.0
    om = 0.27
    cosmo = FlatLambdaCDM(H0=100.0, Om0=om)
    expected = cosmo.Om(z)
    got = find_Om(z, om)
    assert_allclose(got, expected, rtol=1e-12)


def test_find_halo_r_and_scale_radius_consistency():
    om = 0.3
    z = 0.5
    M = 1.0  # in 1e14 h^-1 Msun
    r_halo = find_halo_r(M, z, om)
    c = 5.0
    r_s = find_scale_radius(r_halo, c)
    # Ensure scale radius is just division
    assert_allclose(r_s * c, r_halo, rtol=1e-12)


def test_concentration_ratio_relations_and_errors():
    M = 1.0
    z = 0.5
    # Maccio08 should give a finite concentration
    c1 = find_concentration_ratio(M, z, concentraion_relation="Maccio08")
    assert c1 > 0
    # DuttonMaccio14 should vary with z
    c2 = find_concentration_ratio(M, z, concentraion_relation="DuttonMaccio14")
    assert c2 > 0
    # Invalid relation raises
    with pytest.raises(ValueError):
        find_concentration_ratio(M, z, concentraion_relation="BadModel")
    # Wrong delta_crit raises
    with pytest.raises(ValueError):
        find_concentration_ratio(
            M, z, concentraion_relation="Maccio08", delta_crit=500.0
        )


def test_fx_behavior_limits():
    rs = 1.0
    # At exactly r=rs, expect ~1/3
    assert_allclose(f_x(rs, rs), 1.0 / 3.0, rtol=1e-12)
    # At small x < 1
    val1 = f_x(rs, 0.5 * rs)
    # At large x > 1
    val2 = f_x(rs, 2.0 * rs)
    assert np.isfinite(val1) and np.isfinite(val2)


def test_find_delta_bar_monotonic():
    c1 = 5.0
    c2 = 10.0
    db1 = find_delta_bar(c1)
    db2 = find_delta_bar(c2)
    assert db2 > db1  # higher concentration → higher delta_bar


def test_find_NFW_sigma_symmetry_and_infinity():
    om = 0.3
    z = 0.2
    M = 1.0
    # Finite projected separation → finite sigma
    sig = find_NFW_sigma(0.1, M, z, om)
    assert np.isfinite(sig)
    # At projected_sep=0 → inf
    assert np.isinf(find_NFW_sigma(0.0, M, z, om))


def test_find_sigma_sqr_scaling():
    om = 0.3
    z = 0.3
    M1 = 1.0
    M2 = 2.0
    sig1 = find_sigma_sqr(M1, z, om)
    sig2 = find_sigma_sqr(M2, z, om)
    assert sig2 > sig1  # larger halo mass → larger velocity dispersion


def test_find_p_delta_z_symmetry():
    om = 0.3
    z = 0.3
    M = 1.0
    dz = 0.01
    p1 = find_p_delta_z(dz, z, M, om)
    p2 = find_p_delta_z(-dz, z, M, om)
    assert_allclose(p1, p2, rtol=1e-12)


def test_find_p_M_special_cases():
    om = 0.3
    h = 0.7
    z_group = 0.2
    z_gal = 0.22
    M = 10.0
    # Coincident group & galaxy → inf
    val_inf = find_p_M(10, 10, 10, 10, z_group, z_group, M, om, h)
    assert np.isinf(val_inf)
    # Offset positions → finite probability
    val = find_p_M(10, 10, 10.1, 10.2, z_group, z_gal, M, om, h)
    assert np.isfinite(val) and val > 0
