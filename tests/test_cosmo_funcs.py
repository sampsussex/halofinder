# tests/test_cosmo_utils.py
import numpy as np
import pytest
from numpy.testing import assert_allclose

# Update this import to match where your functions live
from cosmo_funcs import (
    Hubble,
    inverse_hubble,
    comoving_distance,
    get_all_comoving_distance,
    luminosity_distance,
    distance_modulus,
    comoving_volume,
    get_all_comoving_volumes,
    angular_sep,
    find_projected_separation,
    luminosity_to_mag,
    magnitude_to_luminosity,
    get_all_luminosity_to_magnitude,
    get_all_magnitude_to_luminosity,
    spherical_to_cartesian,
    find_all_spherical_to_cartesian,
)

# astropy for reference comparisons
from astropy.cosmology import FlatLambdaCDM

# Choose tolerances for comparisons (some numerical integration error expected)
RTOL = 2e-3  # relative tolerance
ATOL = 1e-6  # absolute tolerance where needed


@pytest.mark.parametrize("z,om", [(0.0, 0.3), (0.1, 0.3), (1.0, 0.3), (2.0, 0.27)])
def test_hubble_matches_astropy(z, om):
    """H(z) should match astropy's FlatLambdaCDM with H0=100 km/s/Mpc (i.e. h=1)."""
    # astropy cosmology with H0 = 100 (so h=1)
    cosmo = FlatLambdaCDM(H0=100.0, Om0=om)
    expected = cosmo.H(z).value  # km / s / Mpc
    got = Hubble(z, om)
    assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("z,om", [(0.0, 0.3), (0.5, 0.3), (1.0, 0.3)])
def test_inverse_hubble(z, om):
    got = inverse_hubble(z, om)
    # Should be reciprocal of Hubble
    assert_allclose(got, 1.0 / Hubble(z, om), rtol=1e-12, atol=0.0)


def test_comoving_distance_array_matches_astropy():
    """Compare comoving distances for a set of redshifts to astropy (H0=100)."""
    zs = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    om = 0.3
    cosmo = FlatLambdaCDM(H0=100.0, Om0=om)  # H0=100 so consistent with functions
    expected = np.array([cosmo.comoving_distance(z).value for z in zs])  # Mpc
    got = get_all_comoving_distance(zs, om)
    # Integration differences -> relax tolerance slightly
    assert_allclose(got, expected, rtol=5e-3, atol=1e-6)


@pytest.mark.parametrize("z,om", [(0.1, 0.3), (0.5, 0.27), (1.0, 0.3)])
def test_luminosity_distance_matches_astropy(z, om):
    """luminosity_distance = (1+z) * comoving_distance; compare to astropy (H0=100)."""
    cosmo = FlatLambdaCDM(H0=100.0, Om0=om)
    expected = cosmo.luminosity_distance(z).value  # Mpc
    got = luminosity_distance(z, om)
    assert_allclose(got, expected, rtol=5e-3, atol=1e-6)


def test_distance_modulus_matches_astropy_for_arbitrary_h():
    """
    distance_modulus(z, omega_matter, h) expects luminosity_distance in Mpc h^-1,
    then divides by 'h' to get Mpc. To compare with astropy run with H0 = 100*h.
    """
    z = 0.5
    om = 0.3
    h = 0.7
    # Astropy cosmology with H0 = 100 * h
    cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=om)
    expected_mu = cosmo.distmod(z).value
    got_mu = distance_modulus(z, om, h)
    assert_allclose(got_mu, expected_mu, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize("z,om", [(0.1, 0.3), (0.5, 0.3)])
def test_comoving_volume_matches_astropy(z, om):
    """Compare comoving volume (4/3 pi d_c^3) with astropy's comoving_volume (H0 scaling)."""
    # Note: the user's comoving_volume uses (4/3) * pi * d_c**3
    # We'll compare that to astropy's comoving_volume with H0=100 so units align.
    cosmo = FlatLambdaCDM(H0=100.0, Om0=om)
    expected = cosmo.comoving_volume(z).value  # Mpc^3
    got = comoving_volume(z, om)
    # astropy's comoving_volume is computed more precisely; allow modest tolerance
    assert_allclose(got, expected, rtol=5e-3, atol=1e-6)


def test_angular_separation_against_astropy():
    """Compare angular separation with astropy.coordinates.SkyCoord.separation."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra1, dec1 = 10.0, -10.0
    ra2, dec2 = 350.0, 20.0  # tests RA wrap-around handling
    got_deg = angular_sep(ra1, dec1, ra2, dec2)
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    expected_deg = c1.separation(c2).deg
    assert_allclose(got_deg, expected_deg, rtol=1e-7, atol=0.0)


def test_projected_separation_consistency():
    """Projected separation = angular separation (radians) * comoving_distance(z)."""
    ra1, dec1 = 10.0, 10.0
    ra2, dec2 = 10.5, 10.2
    z1 = 0.3
    om = 0.3
    ang_rad = np.radians(angular_sep(ra1, dec1, ra2, dec2))
    d_c = comoving_distance(z1, om)
    got = find_projected_separation(ra1, dec1, ra2, dec2, z1, om)
    assert_allclose(got, ang_rad * d_c, rtol=1e-12, atol=0.0)


def test_luminosity_magnitude_roundtrip():
    """Test luminosity <-> magnitude round trip and array helpers."""
    M_sun = 4.5
    L = 2.3  # in units expected by luminosity_to_mag (i.e. 10**14 * h^-1 L_sun)
    mag = luminosity_to_mag(L, M_sun)
    L_back = magnitude_to_luminosity(mag, M_sun)
    # original L vs L_back should match within numerical precision
    assert_allclose(L_back, L * (10**14), rtol=1e-12, atol=0.0)  # magnitude_to_luminosity returns absolute L in L_sun
    # Test array versions
    Ls = np.array([0.1, 1.0, 5.0])
    mags = get_all_luminosity_to_magnitude(Ls, M_sun)
    Ls_back = get_all_magnitude_to_luminosity(mags, M_sun)  # note: this returns L / 1e14
    assert_allclose(Ls_back, Ls, rtol=1e-12, atol=0.0)


def test_spherical_to_cartesian_and_batch():
    ra = 30.0
    dec = -10.0
    d = 1000.0  # Mpc
    xyz = spherical_to_cartesian(ra, dec, d)
    # compute same by numpy
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    expected = np.zeros(3)
    expected[0] = d * np.cos(dec_rad) * np.cos(ra_rad)
    expected[1] = d * np.cos(dec_rad) * np.sin(ra_rad)
    expected[2] = d * np.sin(dec_rad)
    assert_allclose(xyz, expected, rtol=1e-12, atol=0.0)

    # test batch converter
    ras = np.array([ra, ra + 10.0])
    decs = np.array([dec, dec + 5.0])
    ds = np.array([d, d + 200.0])
    all_xyz = find_all_spherical_to_cartesian(ras, decs, ds)
    expected_batch = np.vstack([expected,
                                np.array([
                                    ds[1] * np.cos(np.deg2rad(decs[1])) * np.cos(np.deg2rad(ras[1])),
                                    ds[1] * np.cos(np.deg2rad(decs[1])) * np.sin(np.deg2rad(ras[1])),
                                    ds[1] * np.sin(np.deg2rad(decs[1])),
                                ])])
    assert_allclose(all_xyz, expected_batch, rtol=1e-12, atol=0.0)
