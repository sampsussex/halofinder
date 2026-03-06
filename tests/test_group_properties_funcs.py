import numpy as np
from numpy.testing import assert_allclose

from group_properties_funcs import (
    find_all_initial_mass_to_light,
    find_group_sizes,
    fit_log_luminosity_log_mass_relation,
    brightest_galaxy_centers_fast,
)


def test_find_all_initial_mass_to_light_scales_values():
    luminosity = np.array([1.5, 2.0, 3.5])
    gain = 500.0

    masses = find_all_initial_mass_to_light(luminosity, gain)

    assert_allclose(masses, np.log10(luminosity * gain * 1e14), rtol=0.0, atol=0.0)


def test_find_group_sizes_counts_unique_ids():
    group_ids = np.array([10, 10, 5, 5, 5])

    sizes = find_group_sizes(group_ids)

    assert_allclose(sizes, np.array([3, 2]))


def test_fit_log_luminosity_log_mass_relation_recovers_linear_relation():
    lum = np.array([1.0, 2.0, 4.0, 8.0]) / 1e14
    # log10(M) = 2 * log10(L*1e14) + 12
    masses = 2.0 * np.log10(lum * 1e14) + 12.0
    sizes = np.array([6, 6, 6, 6])

    slope, intercept, n_used = fit_log_luminosity_log_mass_relation(lum, masses, sizes, 5)

    assert_allclose(slope, 2.0, atol=1e-10)
    assert_allclose(intercept, 12.0, atol=1e-10)
    assert n_used == 4


def test_brightest_galaxy_centers_fast_returns_three_biggest_stellar_mass_sum():
    luminosity = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
    stellar_mass = np.array([10.0, 7.0, 2.0, 6.0, 4.0])
    abs_mags = np.array([-21.0, -20.5, -19.0, -20.0, -19.5])
    is_red = np.array([True, False, True, False, True])
    ra = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dec = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    z = np.array([0.1, 0.1, 0.1, 0.2, 0.2])
    group_ids = np.array([10, 10, 10, 20, 20])

    (
        unique_groups,
        _centers_ra,
        _centers_dec,
        _centers_z,
        _centers_lum,
        group_stellar_mass,
        group_stellar_mass_3_biggest,
        _bcg_mag,
        _group_sizes,
        _central_is_red,
    ) = brightest_galaxy_centers_fast(
        luminosity,
        stellar_mass,
        abs_mags,
        is_red,
        ra,
        dec,
        z,
        group_ids,
        0.01,
        -20.0,
        -1.0,
        20.0,
        0.3,
        0.7,
    )

    order = np.argsort(unique_groups)
    unique_groups = unique_groups[order]
    group_stellar_mass = group_stellar_mass[order]
    group_stellar_mass_3_biggest = group_stellar_mass_3_biggest[order]

    assert_allclose(group_stellar_mass, np.array([19.0, 10.0]))
    assert_allclose(group_stellar_mass_3_biggest, np.array([19.0, 10.0]))
