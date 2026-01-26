import numpy as np
from numpy.testing import assert_allclose

from group_properties_funcs import (
    find_all_initial_mass_to_light,
    find_group_sizes,
)


def test_find_all_initial_mass_to_light_scales_values():
    luminosity = np.array([1.5, 2.0, 3.5])
    gain = 500.0

    masses = find_all_initial_mass_to_light(luminosity, gain)

    assert_allclose(masses, luminosity * gain, rtol=0.0, atol=0.0)


def test_find_group_sizes_counts_unique_ids():
    group_ids = np.array([10, 10, 5, 5, 5])

    sizes = find_group_sizes(group_ids)

    assert_allclose(sizes, np.array([3, 2]))
