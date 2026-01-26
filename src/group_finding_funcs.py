# Yang 07/21 groupfinder
import numpy as np
from numba import njit, prange, float64, int64
from cosmo_funcs import get_all_comoving_distance, find_all_spherical_to_cartesian, spherical_to_cartesian
from halo_p_M_funcs import find_p_M

@njit
def negative_exponential_func(x, B_a, B_b, B_c):
    """ A helper function to compute negative exponential values."""
    return B_a * np.exp(-B_b * x) + B_c


@njit(float64[:](int64[:], int64, float64[:], float64[:], float64[:], 
                 float64, float64, float64, float64, float64, float64), parallel=True)
def compute_probabilities_parallel(
    indices, central_idx, galaxy_ra, galaxy_dec, galaxy_z,
    group_ra_i, group_dec_i, group_z_i, group_halo_mass_i, omega_matter, h
):
    """Helper function to compute probabilities in parallel"""
    probs = np.zeros(len(indices), dtype=np.float64)
    
    for j in prange(len(indices)):
        neighbor_idx = indices[j]
        if neighbor_idx != central_idx:
            probs[j] = find_p_M(
                galaxy_ra[neighbor_idx], galaxy_dec[neighbor_idx],
                group_ra_i, group_dec_i,
                group_z_i, galaxy_z[neighbor_idx],
                group_halo_mass_i, omega_matter, h
            )
        else:
            probs[j] = -1.0
    
    return probs


@njit
def update_group_membership_tinker(
    galaxy_ra, galaxy_dec, galaxy_z, galaxy_group_id,
    group_ids, group_ra, group_dec, group_z, group_sizes, group_halo_mass,
    galaxy_tree, is_central, is_satellite, is_red, thresh_red_a, thresh_red_b, thresh_red_c, thresh_blue_a, thresh_blue_b, thresh_blue_c, omega_matter, h
):
    """
    Update galaxy group membership based on probability threshold.
    
    Parameters:
    - galaxy_ra: Right ascension of galaxies in degrees
    - galaxy_dec: Declination of galaxies in degrees
    - galaxy_z: Redshift of galaxies
    - galaxy_group_id: Current group IDs of galaxies
    - group_ids: Unique IDs of groups
    - group_ra: Right ascension of groups in degrees
    - group_dec: Declination of groups in degrees
    - group_z: Redshift of groups
    - group_sizes: Sizes of each group (number of galaxies in each group)
    - group_halo_mass: Halo mass of each group in units of log10(h^-1 M_sun)
    - galaxy_tree: KDTree for galaxy positions in spherical coordinates
    - is_central: input boolean array indicating current central galaxies
    - is_satellite: input boolean array indicating current satellite galaxies
    - is_red: boolean array indicating if galaxies are classified as red
    - thresh_red_a, thresh_red_b, thresh_red_c: Parameters for red galaxy threshold function
    - thresh_blue_a, thresh_blue_b, thresh_blue_c: Parameters for blue galaxy threshold function
    - omega_matter: Matter density parameter at z=0
    - h: Dimensionless Hubble parameter (H0/100)
    
    Returns:
    - updated_galaxy_group_id: numpy array of updated group IDs
    - updated_is_central: boolean array indicating central galaxies
    - updated_is_satellite: boolean array indicating satellite galaxies
    """
    n_galaxies = len(galaxy_ra)
    n_groups = len(group_ids)
    
    # Initialize output arrays as copies of inputs
    updated_galaxy_group_id = galaxy_group_id.copy()
    updated_is_central = is_central.copy()
    updated_is_satellite = is_satellite.copy()
    #changed_satellite = np.zeros(n_galaxies, dtype=np.bool_)
    
    # Create mapping from group_id to group index
    max_group_id = np.max(group_ids)
    group_id_to_idx = np.full(max_group_id + 1, -1, dtype=np.int64)
    for idx in prange(len(group_ids)):
        group_id_to_idx[group_ids[idx]] = idx
    
    # Find which galaxy is the central for each group
    central_galaxy_indices = np.full(n_groups, -1, dtype=np.int64)
    for i in range(n_galaxies):
        if updated_is_central[i] and galaxy_group_id[i] >= 0:
            group_idx = group_id_to_idx[galaxy_group_id[i]]
            if group_idx >= 0:
                central_galaxy_indices[group_idx] = i
    
    # Sort groups by halo mass in descending order
    sorted_group_indices = np.argsort(-group_halo_mass)
    
    # Get group coordinates in cartesian space
    group_comoving_distance = get_all_comoving_distance(group_z, omega_matter)
    group_coords = find_all_spherical_to_cartesian(group_ra, group_dec, group_comoving_distance)
    
    # Pre-compute all probabilities for efficiency
    # This is where we can gain the most from parallelization
    
    
# Process each group in order of decreasing halo mass
    for sorted_idx in range(n_groups):
        group_idx = sorted_group_indices[sorted_idx]
        central_idx = central_galaxy_indices[group_idx]
        
        if central_idx < 0:
            continue
            
        # Skip if this central has been reclassified as satellite
        if updated_is_satellite[central_idx]:
            continue
        
        # Query nearby galaxies
        query_point = group_coords[group_idx]
        k = min(500, n_galaxies)
        distances, indices, extra = galaxy_tree.query(query_point, k=k)
        flat_indices = indices[0]
        
        # Compute probabilities in parallel
        probs = compute_probabilities_parallel(
            flat_indices, central_idx, galaxy_ra, galaxy_dec, galaxy_z,
            group_ra[group_idx], group_dec[group_idx], 
            group_z[group_idx], group_halo_mass[group_idx], omega_matter, h
        )
        
        # Apply membership updates (sequential)
        for j in range(len(flat_indices)):
            neighbor_idx = flat_indices[j]
            prob = probs[j]
            
            if prob < 0:  # This was the central itself
                continue
                
            # Skip if already assigned as satellite of this group
            if updated_galaxy_group_id[neighbor_idx] == group_ids[group_idx]:
                continue

            # Skip if already a satellite of a more massive group
            if updated_is_satellite[neighbor_idx]:
                neighbor_group_idx = group_id_to_idx[updated_galaxy_group_id[neighbor_idx]]
                if neighbor_group_idx >= 0 and group_halo_mass[neighbor_group_idx] > group_halo_mass[group_idx]:
                    continue

                
            # Skip if it's a central of a more massive group 
            if updated_is_central[neighbor_idx]:
                neighbor_group_idx = group_id_to_idx[updated_galaxy_group_id[neighbor_idx]]
                if neighbor_group_idx >= 0 and group_halo_mass[neighbor_group_idx] >= group_halo_mass[group_idx]:
                    continue

            
            
            # Assign as satellite if probability exceeds threshold
            if is_red[neighbor_idx]:
                threshold = negative_exponential_func(
                    group_sizes[group_idx], thresh_red_a, thresh_red_b, thresh_red_c
                )
            else:
                threshold = negative_exponential_func(
                    group_sizes[group_idx], thresh_blue_a, thresh_blue_b, thresh_blue_c
                )
                
            if prob > threshold:
                updated_galaxy_group_id[neighbor_idx] = group_ids[group_idx]
                updated_is_satellite[neighbor_idx] = True
                # If it was previously a central, mark it as no longer central
                if updated_is_central[neighbor_idx]:
                    updated_is_central[neighbor_idx] = False
    
    return updated_galaxy_group_id, updated_is_central, updated_is_satellite


@njit
def kdtree_fof(gal_ids, comoving_distance, ra, dec, ll_proj, kdtree):
    """
    Friends-of-friends finder using Numba KD-tree for efficient neighbor searching.
    
    Parameters:
    -----------
    gal_ids : array
        Galaxy IDs
    comoving_distance : array
        Observed comoving distances.
    ra : array
        Right ascension coordinates
    dec : array
        Declination coordinates
    ll_proj : float
        Linking length (projected separation threshold)
    kdtree : numba KDTree
        Pre-built KD-tree for spatial queries
    coords : array
        2D array of coordinates used to build the KD-tree (shape: n_points x n_dimensions)
        Should match the coordinate system used for ll_proj
    
    Returns:
    --------
    group_ids : array
        Group assignment for each galaxy (-1 if unassigned)
    """
    n = len(gal_ids)
    group_ids = -1 * np.ones(n, dtype=np.int64)
    group_counter = 0

    for i in range(n):
        if group_ids[i] != -1:
            continue  # Already assigned

        # Start a new group
        group_ids[i] = group_counter
        to_check = [i]

        while len(to_check) > 0:
            current = to_check.pop()
            
            # Query KD-tree for neighbors within linking length
            # Note: query_radius expects a 2D array, so we reshape current point
            current_coords = spherical_to_cartesian(ra[current], dec[current], comoving_distance[current])
            #print(current_coords)  # Shape: (1, n_dims)
            neighbor_indices = kdtree.query_radius(current_coords, r=ll_proj, return_sorted=True)
            
            # neighbor_indices is a list with one element (for the single query point)
            neighbors = neighbor_indices[0]
            
            for neighbor_idx in neighbors:
                if group_ids[neighbor_idx] != -1:
                    continue  # Already assigned
                
                # Additional redshift check if needed (uncomment if required)
                # if abs(zobs[current] - zobs[neighbor_idx]) > redshift_threshold:
                #     continue
                
                group_ids[neighbor_idx] = group_counter
                to_check.append(neighbor_idx)

        group_counter += 1

    return group_ids
