import numpy as np
from numba import njit, prange
from luminosity_mass_funcs import luminosity_correction_factor
from cosmo_funcs import (luminosity_to_mag,
                         magnitude_to_luminosity,
                         comoving_distance,
                         spherical_to_cartesian,
                         get_all_comoving_distance,
                         find_all_spherical_to_cartesian,
                         get_all_luminosity_to_magnitude,
                         get_all_magnitude_to_luminosity,
                         cartesian_to_spherical
                         )
from typing import Optional, Sequence, Tuple, List

@njit
def find_all_initial_mass_to_light(group_luminosity, mass_light_gain):
    """Calculate initial halo masses based on group luminosity and mass-light gain factor."""
    # placeholder, using intitaly 500 value
    halo_masses = np.zeros(len(group_luminosity))
    for i in range(len(group_luminosity)):
        halo_masses[i] = mass_light_gain * group_luminosity[i]
    return halo_masses


@njit
def find_group_sizes(group_ids):
    """Returns the number of galaxies in each group from group IDs."""
    unique_ids = np.unique(group_ids)
    remapped = np.searchsorted(unique_ids, group_ids)
    sizes = np.bincount(remapped)
    return sizes


@njit(parallel=True)
def brightest_galaxy_centers(
    luminosity,
    abs_mags,
    is_red,
    ra,
    dec,
    z,
    group_ids,
    phi_star,
    M_star,
    alpha,
    mag_limit,
    omega_matter,
    h,
):
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

            L_corr = luminosity_correction_factor(
                mag_limit, Z[0], phi_star, M_star, alpha, omega_matter, h
            )
            # print(L_corr)
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

            L_corr = luminosity_correction_factor(
                mag_limit, Z[brightest_idx], phi_star, M_star, alpha, omega_matter, h
            )
            # print(L_corr)
            centers_lum[i] = Lsum * L_corr
            central_is_red[i] = is_red[mask][brightest_idx]

            bcg_mag[i] = abs_mags[mask][brightest_idx]

    return (
        unique_groups,
        centers_ra,
        centers_dec,
        centers_z,
        centers_lum,
        bcg_mag,
        group_sizes,
        central_is_red,
    )


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
    order = np.argsort(group_ids)  # JIT-supported
    gid_sorted = group_ids[order]

    # Edge case: empty input
    if n == 0:
        return (
            order,
            gid_sorted,
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty(0, np.int64),
        )

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
    order,
    unique_gids,
    starts,
    ends,
    luminosity,
    stellar_mass,
    abs_mags,
    is_red,
    ra,
    dec,
    z,
    phi_star,
    M_star,
    alpha,
    mag_limit,
    omega_matter,
    h,
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
            j = order[k]  # original galaxy index
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

    return (
        unique_gids,
        centers_ra,
        centers_dec,
        centers_z,
        centers_lum,
        group_stellar_mass,
        bcg_mag,
        group_sizes,
        central_is_red,
    )


@njit
def brightest_galaxy_centers_fast(
    luminosity,
    stellar_mass,
    abs_mags,
    is_red,
    ra,
    dec,
    z,
    group_ids,
    phi_star,
    M_star,
    alpha,
    mag_limit,
    omega_matter,
    h,
):
    # (Recommended) ensure contiguous dtypes once outside the JIT hot path

    order, gid_sorted, unique_gids, starts, ends = sort_and_build_segments(group_ids)

    return brightest_galaxy_centers_from_segments(
        order,
        unique_gids,
        starts,
        ends,
        luminosity,
        stellar_mass,
        abs_mags,
        is_red,
        ra,
        dec,
        z,
        phi_star,
        M_star,
        alpha,
        mag_limit,
        omega_matter,
        h,
    )
# -----------------------------
# Numba helpers: stats
# -----------------------------
@njit(cache=True)
def mean_1d(x: np.ndarray) -> float:
    s = 0.0
    n = x.size
    for i in range(n):
        s += x[i]
    return s / n if n > 0 else np.nan


@njit(cache=True)
def median_sorted(x_sorted: np.ndarray) -> float:
    n = x_sorted.size
    if n == 0:
        return np.nan
    mid = n // 2
    if n % 2 == 1:
        return x_sorted[mid]
    return 0.5 * (x_sorted[mid - 1] + x_sorted[mid])


@njit(cache=True)
def median_1d(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    xs = np.sort(x.copy())
    return median_sorted(xs)


@njit(cache=True)
def quantile_interpolated_sorted(x_sorted: np.ndarray, q: float) -> float:
    """
    Linear-interpolated quantile on already-sorted array (like many "type=7" defs).
    """
    n = x_sorted.size
    if n == 0:
        return np.nan
    if q <= 0.0:
        return x_sorted[0]
    if q >= 1.0:
        return x_sorted[n - 1]

    # position in [0, n-1]
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if hi == lo:
        return x_sorted[lo]
    t = pos - lo
    return (1.0 - t) * x_sorted[lo] + t * x_sorted[hi]


@njit(cache=True)
def quantile_interpolated(x: np.ndarray, q: float) -> float:
    xs = np.sort(x.copy())
    return quantile_interpolated_sorted(xs, q)


@njit(cache=True)
def euclidean_distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


@njit(cache=True)
def velocity_dispersion_gapper(redshifts: np.ndarray, vel_errs: np.ndarray) -> Tuple[float, float]:
    """
    Returns: (dispersion, sigma_err)
    Implements the same logic as the Rust code.
    """
    n = redshifts.size
    if n < 2:
        return 0.0, np.sqrt(mean_1d(vel_errs)) if vel_errs.size > 0 else 0.0

    sigma_err_sq = mean_1d(vel_errs)
    z_med = median_1d(redshifts)

    # v_i = (z_i * c) / (1 + z_med)
    velocities = np.empty(n, dtype=np.float64)
    denom = 1.0 + z_med
    C = 299792.458
    for i in range(n):
        velocities[i] = (redshifts[i] * C) / denom

    velocities.sort()

    # gaps (n-1)
    gaps = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        gaps[i] = velocities[i + 1] - velocities[i]

    # weights: i*(n-i) for i=1..n-1
    # sum weights*gaps
    s = 0.0
    nf = float(n)
    for i in range(1, n):
        w = i * (n - i)
        s += float(w) * gaps[i - 1]

    sigma_gap = (np.sqrt(np.pi) / (nf * (nf - 1.0))) * s
    raw_disp_sq = (nf * sigma_gap * sigma_gap) / (nf - 1.0)

    if raw_disp_sq > sigma_err_sq:
        disp = np.sqrt(raw_disp_sq - sigma_err_sq)
    else:
        disp = 0.0

    return disp, np.sqrt(sigma_err_sq)


@njit(cache=True)
def calculate_iterative_center_idx(ra_deg: np.ndarray, dec_deg: np.ndarray, mags: np.ndarray, M_sun: np.float64) -> int:
    """
    Returns original index (0..n-1) of the final remaining object after iteratively
    removing the furthest object from the flux-weighted center (in cartesian space),
    then choosing the remaining with highest flux.
    """
    n = ra_deg.size
    if n == 0:
        return -1
    if n == 1:
        return 0

    # coords unit vectors
    coords = np.empty((n, 3), dtype=np.float64)
    flux = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = spherical_to_cartesian(ra_deg[i], dec_deg[i], 1.0)
        coords[i, 0] = v[0]
        coords[i, 1] = v[1]
        coords[i, 2] = v[2]
        flux[i] = magnitude_to_luminosity(mags[i], M_sun)

    alive = np.ones(n, dtype=np.uint8)  # 1=keep,0=removed
    alive_count = n

    # while > 2 alive
    while alive_count > 2:
        # flux sum
        fsum = 0.0
        for i in range(n):
            if alive[i] == 1:
                fsum += flux[i]
        if fsum == 0.0:
            break

        # center = sum(coord*flux)/fsum
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for i in range(n):
            if alive[i] == 1:
                f = flux[i]
                cx += coords[i, 0] * f
                cy += coords[i, 1] * f
                cz += coords[i, 2] * f
        cx /= fsum
        cy /= fsum
        cz /= fsum

        # find furthest alive
        max_d = -1.0
        max_i = -1
        for i in range(n):
            if alive[i] == 1:
                dx = coords[i, 0] - cx
                dy = coords[i, 1] - cy
                dz = coords[i, 2] - cz
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d > max_d:
                    max_d = d
                    max_i = i

        if max_i < 0:
            break

        alive[max_i] = 0
        alive_count -= 1

    # among remaining, choose highest flux
    best_i = -1
    best_f = -1.0
    for i in range(n):
        if alive[i] == 1:
            if flux[i] > best_f:
                best_f = flux[i]
                best_i = i
    return best_i

@njit
def calculate_radius(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    group_center_ra: float,
    group_center_dec: float,
    group_center_z: float,
    omega_matter: float,
) -> np.ndarray:
    """
    Returns [R50, R68, R100] in Mpc (comoving, using center distance).
    """
    n = ra_deg.size
    if n == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    dist = float(comoving_distance(group_center_z, omega_matter))  # Mpc
    center = spherical_to_cartesian(group_center_ra, group_center_dec, dist)

    dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        pos = spherical_to_cartesian(ra_deg[i], dec_deg[i], dist)
        dists[i] = euclidean_distance_3d(pos, center)

    dists.sort()

    q50 = quantile_interpolated_sorted(dists, 0.5)
    q68 = quantile_interpolated_sorted(dists, 0.68)
    q100 = dists[-1]
    return np.array([q50, q68, q100], dtype=np.float64)


@njit(cache=True)
def calculate_center_of_light(ra_deg: np.ndarray, dec_deg: np.ndarray, mags: np.ndarray, M_sun: np.float64) -> Tuple[float, float]:
    n = ra_deg.size
    if n == 0:
        return np.nan, np.nan

    fluxes = np.empty(n, dtype=np.float64)
    sum_flux = 0.0
    for i in range(n):
        f = magnitude_to_luminosity(mags[i], M_sun)
        fluxes[i] = f
        sum_flux += f
    if sum_flux == 0.0:
        return np.nan, np.nan

    wx = 0.0
    wy = 0.0
    wz = 0.0
    for i in range(n):
        v = spherical_to_cartesian(ra_deg[i], dec_deg[i])
        f = fluxes[i]
        wx += v[0] * f
        wy += v[1] * f
        wz += v[2] * f

    wx /= sum_flux
    wy /= sum_flux
    wz /= sum_flux

    eq = spherical_to_cartesian(wx, wy, wz)
    return float(eq[0]), float(eq[1])


@njit(cache=True)
def calculate_flux_weighted_redshift(redshifts: np.ndarray, mags: np.ndarray, M_sun: np.float64) -> float:
    n = redshifts.size
    if n == 0:
        return np.nan
    sum_flux = 0.0
    num = 0.0
    for i in range(n):
        f = magnitude_to_luminosity(mags[i], M_sun)
        sum_flux += f
        num += redshifts[i] * f
    return num / sum_flux if sum_flux != 0.0 else np.nan


@njit(cache=True)
def calculate_total_mass(gravitational_radius_mpc: float, los_velocity_dispersion_kms: float) -> float:
    """
    Tempel+2014 Eqn 8 style (as given).
    """
    return 2.325e12 * gravitational_radius_mpc * ((3.0 ** (1.0 / 3.0)) * los_velocity_dispersion_kms / 100.0) ** 2


@njit
def calculate_velocity_disp_corr_mass(
    radius_mpc: float,
    los_velocity_dispersion_kms: float,
    cosmo,
) -> float:
    """
    van Kampen+2026 style correction. Kept as Python because it needs cosmo.h0.
    """
    alpha = 1.030
    dispersion_limit = 244.634
    n1 = -1.989
    beta = 0.213
    rad_lim = 0.369 * 0.7 / (cosmo.h0 / 100.0)
    n2 = -1.591

    a_b = 0.0
    a_c = 0.0
    if los_velocity_dispersion_kms < dispersion_limit:
        a_b = alpha * ((los_velocity_dispersion_kms / dispersion_limit) ** n1 - 1.0)
    if radius_mpc < rad_lim:
        a_c = beta * ((radius_mpc / rad_lim) ** n2 - 1.0)

    correction_factor = 5.0 / 3.0 + a_b + a_c
    G_MSOL_MPC_KMS2 = 4.302e-9 #Mpc (km/s)^2 / M_sun
    return correction_factor * (los_velocity_dispersion_kms**2) * radius_mpc / G_MSOL_MPC_KMS2


@njit(cache=True)
def dynamical_mass(gapper_velocity_dispersion, r50, A):
    G_MSOL_MPC_KMS2 = 4.302e-9 #Mpc (km/s)^2 / M_sun
    raw_mass = (r50 * (gapper_velocity_dispersion**2)) / G_MSOL_MPC_KMS2 if np.isfinite(r50) else np.nan
    return A * raw_mass if np.isfinite(raw_mass) else np.nan


@njit(parallel=True)
def fit_log_luminosity_log_mass_relation(group_luminosities, group_dynamical_masses, group_sizes, min_group_members):
    """
    Fit log10(Mdyn) = intercept + slope * log10(Lgroup) using groups above a size threshold.
    Returns (slope, intercept, n_used).
    """
    n = group_luminosities.size
    valid_count = 0
    for i in range(n):
        if group_sizes[i] >= min_group_members and group_luminosities[i] > 0.0 and group_dynamical_masses[i] > 0.0:
            valid_count += 1

    if valid_count < 2:
        return np.nan, np.nan, valid_count

    x = np.empty(valid_count, dtype=np.float64)
    y = np.empty(valid_count, dtype=np.float64)

    j = 0
    for i in range(n):
        if group_sizes[i] >= min_group_members and group_luminosities[i] > 0.0 and group_dynamical_masses[i] > 0.0:
            x[j] = np.log10(group_luminosities[i] * 1e14)
            y[j] = np.log10(group_dynamical_masses[i])
            j += 1

    mean_x = 0.0
    mean_y = 0.0
    for i in range(valid_count):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= valid_count
    mean_y /= valid_count

    var_x = 0.0
    cov_xy = 0.0
    for i in range(valid_count):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        var_x += dx * dx
        cov_xy += dx * dy

    if var_x <= 0.0:
        return np.nan, np.nan, valid_count

    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept, valid_count


@njit(parallel=True)
def calculate_group_dynamical_masses(group_ids, unique_groups, zobs, ra, dec, group_centres_ra, group_centres_dec, group_centres_z, group_sizes, A, omega_matter):
    """Compute dynamical mass per group from member redshifts and R50 proxy radius."""
    n_groups = unique_groups.size
    masses = np.empty(n_groups, dtype=np.float64)
    vel_errs = np.zeros(zobs.size, dtype=np.float64)

    for i in prange(n_groups):
        gid = unique_groups[i]
        count = group_sizes[i]
        if count < 2:
            masses[i] = np.nan
            continue

        member_z = np.empty(count, dtype=np.float64)
        member_ra = np.empty(count, dtype=np.float64)
        member_dec = np.empty(count, dtype=np.float64)
        idx = 0
        for j in range(group_ids.size):
            if group_ids[j] == gid:
                member_z[idx] = zobs[j]
                member_ra[idx] = ra[j]
                member_dec[idx] = dec[j]
                idx += 1

        sigma, _ = velocity_dispersion_gapper(member_z, vel_errs[:count])

        radii = calculate_radius(
            member_ra,
            member_dec,
            group_centres_ra[i],
            group_centres_dec[i],
            group_centres_z[i],
            omega_matter,
        )
        r50 = radii[0]

        masses[i] = dynamical_mass(sigma, r50, A)

    return masses
