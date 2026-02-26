import numpy as np
from numba import njit, prange
from luminosity_funcs import luminosity_correction_factor


@njit
def find_all_initial_mass_to_light(group_luminosity, mass_light_gain):
    """ Calculate initial halo masses based on group luminosity and mass-light gain factor."""
    #placeholder, using intitaly 500 value
    halo_masses = np.zeros(len(group_luminosity))
    for i in range(len(group_luminosity)):
        halo_masses[i] = mass_light_gain*group_luminosity[i]
    return halo_masses
    

@njit
def find_group_sizes(group_ids):
    """ Returns the number of galaxies in each group from group IDs."""
    unique_ids = np.unique(group_ids)
    remapped = np.searchsorted(unique_ids, group_ids)
    sizes = np.bincount(remapped)
    return sizes


@njit(parallel=True)
def brightest_galaxy_centers(luminosity, abs_mags, is_red, ra, dec, z, group_ids, phi_star, M_star, alpha, mag_limit, omega_matter, h):
    """
    Returns the group properties and locations from a list of galaxies and group assignments.
    Centers are placed at the brightest galaxy in each group.
    
    Parameters:
    ----------
    luminosity : np.array(float)
        Array of uncorrected galaxy luminosities
    abs_mags : np.array(float)
        Array of absolute magnitudes of galaxies
    is_red : np.array(bool)
        Numpy array indicating if galaxy is classified as red
    ra : np.array(float)
        Right ascension in degrees of galaxies
    dec : np.array(float)
        Declination in degrees of galaxies
    z : np.array(float)
        Redshift of galaxies
    group_ids : np.array(int)
        Numpy array of group IDs
    is_red : np.array(bool)
        Numpy array indicating if galaxy is classified as red
    phi_star : float
        Phi_star parameter from schechter function of galaxy population
    M_star : float
        M_star parameter from schechter function of galaxy population
    alpha : float
        alpha parameter from schechter function of galaxy population
    mag_limit : float
        Apparent magnitude limit of survey used for galaxy selection.
    omega_matter : float
        Matter density parameter at z=0
    h : float
        Dimensionless Hubble parameter (H0/100)
    
    Returns:
    -------
    np.array(int)
        Unique groups in membership assignments
    np.array(float)
        Right ascension of unique group centers in degrees
    np.array(float)
        Declination of unique group centers in degrees
    np.array(float)
        Redshift of unique group centers
    np.array(float)
        Corrected total luminosity of each unique group
    np.array(int)
        Number galaxy members in each unique group
    np.array(bool)
        Boolean array indicating if the brightest galaxy in each group is red
    """
    unique_groups = np.unique(group_ids)
    n_groups = unique_groups.shape[0]
    centers_ra = np.zeros(n_groups)
    centers_dec = np.zeros(n_groups)
    centers_z = np.zeros(n_groups)
    centers_lum = np.zeros(n_groups)
    group_sizes = find_group_sizes(group_ids)
    bcg_mag = np.zeros(n_groups)
    central_is_red = np.zeros(n_groups, dtype=np.bool_)
    
    for i in prange(n_groups):
        gid = unique_groups[i]
        # Select galaxies in group
        mask = group_ids == gid
        
        # Extract data for group
        L = luminosity[mask]
        RA = ra[mask]
        DEC = dec[mask]
        Z = z[mask]
        
        # Check if group has only one galaxy
        if len(L) == 1:
            # Single galaxy case - use its properties directly
            centers_ra[i] = RA[0]
            centers_dec[i] = DEC[0]
            centers_z[i] = Z[0]
            # Still need luminosity correction
            
            L_corr = luminosity_correction_factor(mag_limit, Z[0], phi_star, M_star, alpha, omega_matter, h)
            #print(L_corr)
            centers_lum[i] = L[0] * L_corr

            bcg_mag[i] = abs_mags[mask][0]


            central_is_red[i] = is_red[mask][0]

        else:
            # Multi-galaxy case - find brightest galaxy
            brightest_idx = np.argmax(L)
            
            # Use brightest galaxy's position and redshift as center
            centers_ra[i] = RA[brightest_idx]
            centers_dec[i] = DEC[brightest_idx]
            centers_z[i] = Z[brightest_idx]
            
            # Total luminosity with correction based on brightest galaxy's redshift
            Lsum = np.sum(L)
            
            L_corr = luminosity_correction_factor(mag_limit, Z[brightest_idx], phi_star, M_star, alpha, omega_matter, h)
            #print(L_corr)
            centers_lum[i] = Lsum * L_corr
            central_is_red[i] = is_red[mask][brightest_idx]


            bcg_mag[i] = abs_mags[mask][brightest_idx]
    
    return unique_groups, centers_ra, centers_dec, centers_z, centers_lum, bcg_mag, group_sizes, central_is_red


@njit(cache=True)
def sort_and_build_segments(group_ids):
    """
    Returns:
      order        : indices that sort group_ids
      gid_sorted   : group_ids[order]
      unique_gids  : unique group id per segment (length n_groups)
      starts       : start index (in sorted arrays) per group
      ends         : end index (exclusive, in sorted arrays) per group
    """
    n = group_ids.size
    order = np.argsort(group_ids)                 # JIT-supported
    gid_sorted = group_ids[order]

    # Edge case: empty input
    if n == 0:
        return order, gid_sorted, np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)

    # Pass 1: count groups (= number of runs)
    n_groups = 1
    prev = gid_sorted[0]
    for i in range(1, n):
        g = gid_sorted[i]
        if g != prev:
            n_groups += 1
            prev = g

    # Allocate segment arrays
    starts = np.empty(n_groups, np.int64)
    ends = np.empty(n_groups, np.int64)
    unique_gids = np.empty(n_groups, np.int64)

    # Pass 2: fill segments
    gi = 0
    starts[gi] = 0
    unique_gids[gi] = gid_sorted[0]
    prev = gid_sorted[0]

    for i in range(1, n):
        g = gid_sorted[i]
        if g != prev:
            ends[gi] = i
            gi += 1
            starts[gi] = i
            unique_gids[gi] = g
            prev = g

    ends[gi] = n
    return order, gid_sorted, unique_gids, starts, ends


# ------------------------------------------------------------
# 2) JIT: compute BCG-centred group properties from segments
# ------------------------------------------------------------
@njit(parallel=True)
def brightest_galaxy_centers_from_segments(
    order, unique_gids, starts, ends,
    luminosity, stellar_mass, abs_mags, is_red, ra, dec, z,
    phi_star, M_star, alpha, mag_limit, omega_matter, h
):
    """
    Same outputs as your original function, but uses (order, starts, ends)
    so each group is a contiguous slice in the sorted index space.
    """
    n_groups = unique_gids.size

    centers_ra = np.empty(n_groups, np.float64)
    centers_dec = np.empty(n_groups, np.float64)
    centers_z = np.empty(n_groups, np.float64)
    centers_lum = np.empty(n_groups, np.float64)
    group_stellar_mass = np.empty(n_groups, np.float64)
    bcg_mag = np.empty(n_groups, np.float64)
    central_is_red = np.empty(n_groups, np.bool_)
    group_sizes = np.empty(n_groups, np.int64)

    for i in prange(n_groups):
        s = starts[i]
        e = ends[i]
        group_sizes[i] = e - s

        # One pass: sum L + find brightest (track original index best_j)
        Lsum = 0.0
        stellar_mass_sum = 0.0
        best_L = -1.0e300
        best_j = -1

        for k in range(s, e):
            j = order[k]               # original galaxy index
            Lj = luminosity[j]
            Lsum += Lj
            stellar_mass_sum += stellar_mass[j]
            if Lj > best_L:
                best_L = Lj
                best_j = j

        # Center at brightest galaxy
        cz = z[best_j]
        centers_ra[i] = ra[best_j]
        centers_dec[i] = dec[best_j]
        centers_z[i] = cz
        bcg_mag[i] = abs_mags[best_j]
        central_is_red[i] = is_red[best_j]

        # Luminosity correction evaluated at BCG z (as in your original)
        L_corr = luminosity_correction_factor(
            mag_limit, cz, phi_star, M_star, alpha, omega_matter, h
        )
        centers_lum[i] = Lsum * L_corr
        group_stellar_mass[i] = stellar_mass_sum

    return unique_gids, centers_ra, centers_dec, centers_z, centers_lum, group_stellar_mass, bcg_mag, group_sizes, central_is_red

@njit
def brightest_galaxy_centers_fast(
    luminosity, stellar_mass, abs_mags, is_red, ra, dec, z, group_ids,
    phi_star, M_star, alpha, mag_limit, omega_matter, h
):
    # (Recommended) ensure contiguous dtypes once outside the JIT hot path

    order, gid_sorted, unique_gids, starts, ends = sort_and_build_segments(group_ids)

    return brightest_galaxy_centers_from_segments(
        order, unique_gids, starts, ends,
        luminosity, stellar_mass, abs_mags, is_red, ra, dec, z,
        phi_star, M_star, alpha, mag_limit, omega_matter, h
    )
