import numpy as np
from numpy.testing import assert_allclose

from group_properties_funcs import (
    mean_1d,
    median_sorted,
    median_1d,
    quantile_interpolated_sorted,
    quantile_interpolated,
    euclidean_distance_3d,
    velocity_dispersion_gapper,
    calculate_iterative_center_idx,
    calculate_radius,
    calculate_flux_weighted_redshift,
    calculate_total_mass,
    calculate_velocity_disp_corr_mass,
    dynamical_mass,
    calculate_group_dynamical_masses,
)


class DummyCosmo:
    h0 = 70.0


def test_basic_stat_helpers_cover_edge_paths():
    assert np.isnan(mean_1d(np.array([], dtype=np.float64)))
    assert mean_1d(np.array([1.0, 2.0, 3.0])) == 2.0

    assert np.isnan(median_sorted(np.array([], dtype=np.float64)))
    assert median_sorted(np.array([1.0, 2.0, 3.0])) == 2.0
    assert median_sorted(np.array([1.0, 2.0, 3.0, 4.0])) == 2.5

    assert np.isnan(median_1d(np.array([], dtype=np.float64)))
    assert median_1d(np.array([4.0, 2.0, 1.0])) == 2.0


def test_quantile_helpers_and_distance():
    xs = np.array([1.0, 2.0, 4.0, 8.0])
    x_sorted = np.sort(xs)

    assert quantile_interpolated_sorted(x_sorted, -0.5) == 1.0
    assert quantile_interpolated_sorted(x_sorted, 1.5) == 8.0
    assert quantile_interpolated_sorted(x_sorted, 0.5) == 3.0
    assert quantile_interpolated(xs, 0.25) == 1.75

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 2.0])
    assert euclidean_distance_3d(a, b) == 3.0


def test_velocity_dispersion_gapper_small_n_and_regular_case():
    disp, err = velocity_dispersion_gapper(
        np.array([0.1], dtype=np.float64), np.array([4.0], dtype=np.float64)
    )
    assert disp == 0.0
    assert err == 2.0

    redshifts = np.array([0.10, 0.11, 0.12, 0.13], dtype=np.float64)
    vel_errs = np.zeros_like(redshifts)
    disp2, err2 = velocity_dispersion_gapper(redshifts, vel_errs)
    assert disp2 > 0.0
    assert err2 == 0.0


def test_iterative_center_and_radius_helpers():
    mags = np.array([-21.0, -20.0, -19.0])

    assert calculate_iterative_center_idx(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        4.63,
    ) == -1

    assert calculate_iterative_center_idx(
        np.array([10.0]), np.array([0.0]), np.array([-21.0]), 4.63
    ) == 0

    idx = calculate_iterative_center_idx(
        np.array([10.0, 11.0, 30.0]),
        np.array([0.0, 0.0, 0.0]),
        mags,
        4.63,
    )
    assert idx in (0, 1)

    radii_empty = calculate_radius(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        group_center_ra=10.0,
        group_center_dec=0.0,
        group_center_z=0.1,
        omega_matter=0.3,
    )
    assert np.all(np.isnan(radii_empty))


def test_mass_and_redshift_related_helpers():
    redshifts = np.array([0.1, 0.2])
    mags = np.array([-21.0, -20.0])

    z_flux = calculate_flux_weighted_redshift(redshifts, mags, 4.63)
    assert 0.1 <= z_flux <= 0.2

    total_mass = calculate_total_mass(0.5, 300.0)
    assert total_mass > 0.0

    corr_mass = calculate_velocity_disp_corr_mass(0.2, 100.0, DummyCosmo())
    assert corr_mass > 0.0

    assert np.isnan(dynamical_mass(100.0, np.nan, 1.0))
    assert np.isnan(dynamical_mass(100.0, 0.1, 0.0))
    assert np.isfinite(dynamical_mass(100.0, 0.1, 1.0))


def test_calculate_group_dynamical_masses_includes_single_member_nan():
    group_ids = np.array([10, 10, 20])
    unique_groups = np.array([10, 20])
    zobs = np.array([0.1, 0.11, 0.2])
    ra = np.array([10.0, 10.2, 20.0])
    dec = np.array([0.0, 0.1, 1.0])
    group_centres_ra = np.array([10.1, 20.0])
    group_centres_dec = np.array([0.05, 1.0])
    group_centres_z = np.array([0.105, 0.2])
    group_sizes = np.array([2, 1])

    masses = calculate_group_dynamical_masses(
        group_ids,
        unique_groups,
        zobs,
        ra,
        dec,
        group_centres_ra,
        group_centres_dec,
        group_centres_z,
        group_sizes,
        A=1.0,
        omega_matter=0.3,
    )

    assert masses.shape == unique_groups.shape
    assert np.isfinite(masses[0])
    assert np.isnan(masses[1])
