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
    bcg_mag = np.zeros(n_groups, dtype=np.int64)
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
