import numpy as np
from numpy.testing import assert_allclose

from group_properties_funcs import (
    find_all_initial_mass_to_light,
    find_group_sizes,
    fit_log_luminosity_log_mass_relation,
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
