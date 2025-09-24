import numpy as np
from numba import njit
from astropy.constants import c


@njit
def Hubble(z, omega_matter):
    """ Get the Hubble parameter at redshift z. Flat LCDM Cosmology specified in global variables. In units of km/s/Mpc.
    Parameters:
        z (float): redshift
        omega_matter (float): Matter density parameter at z=0
        Returns:
        float: Hubble parameter at redshift z in km/s/Mpc h^-1
        """
    omega_lambda = 1. - omega_matter
    return 100. * np.sqrt(omega_matter * (1. + z)**3 + omega_lambda)


@njit
def inverse_hubble(z, omega_matter):
    """ Get the inverse Hubble parameter at redshift z. Flat LCDM Cosmology specified in global variables. In units of Mpc/km/s.
    Parameters:
        z (float): redshift
        omega_matter (float): Matter density parameter at z=0
        Returns:
        float: Inverse Hubble parameter at redshift z in Mpc/km/s h^-1
    """
    
    return 1.0 / Hubble(z, omega_matter)


@njit
def simpson_integrate_inv_hubble(func, a, b, n, omega_matter):
    if n % 2 == 1:
        n += 1
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    
    for i in range(n + 1):
        y[i] = func(x[i], omega_matter)   # pass omega_matter explicitly
    
    result = y[0] + y[n]
    
    for i in range(1, n, 2):
        result += 4 * y[i]
    
    for i in range(2, n, 2):
        result += 2 * y[i]
    
    return result * h / 3


@njit
def comoving_distance(z, omega_matter):
    """
    Get the comoving distance at redshift z. Flat LCDM Cosmology specified in global variables.


    Parameters:
        z (float): redshift
        omega_matter (float): Matter density parameter at z=0

    Returns:
        float: Comoving distance in Mpc h^-1
    """
    C = 299792.458 # km/s
    integral_result = simpson_integrate_inv_hubble(inverse_hubble, 0., z, 1000, omega_matter)
    return C * integral_result


@njit
def get_all_comoving_distance(z_array, omega_matter):
    """
    Get all the comoving distance at redshift z of an array. Flat LCDM Cosmology specified in global variables.


    Parameters:
        z array(float): redshift
        omega_matter (float): Matter density parameter at z=0

    Returns:
        array(float): Comoving distance in Mpc
    """

    dms = np.zeros(len(z_array))
    for i in range(len(z_array)):
        dms[i] = comoving_distance(z_array[i], omega_matter)
    return dms


@njit
def luminosity_distance(z, omega_matter):
    """
    Get the luminosity distance at redshift z. Flat LCDM Cosmology specified in global variables.

    Parameters:
        z (float): redshift
        omega_matter (float): Matter density parameter at z=0

    Returns:
        float: Luminosity distance in Mpc h^-1
    """
    return (1 + z) * comoving_distance(z, omega_matter)  # in Mpc h^-1


@njit
def distance_modulus(z, omega_matter, h):
    """
    Compute the distance modulus at redshift z for a flat LCDM cosmology.

    Assumes luminosity_distance(z) returns distances in Mpc h^-1
    and uses global variable h (dimensionless Hubble parameter).

    Parameters:
        z (float): redshift
        omega_matter (float): Matter density parameter at z=0
        h (float): Dimensionless Hubble parameter (H0 / 100 km/s/Mpc)

    Returns:
        float: Distance modulus (mag)
    """
    Dl_Mpc = luminosity_distance(z, omega_matter) / h  # convert from Mpc h^-1 to Mpc
    mu = 5.0 * np.log10(Dl_Mpc) + 25.0
    return mu


@njit
def comoving_volume(z, omega_matter):
    """
    Get the comoving volume out to redshift z. Flat LCDM Cosmology specified in global variables.


    Parameters:
        z (float): redshift
        omega_matter (float): Matter density parameter at z=0

    Returns:
        float: Comoving volume in Mpc^3 h^-3
    """
    d_c = comoving_distance(z, omega_matter)  # in Mpc
    return (4. / 3.) * np.pi * d_c**3  # in Mpc^3 h^-3


@njit
def get_all_comoving_volumes(zs, omega_matter):
    """
    Get all the comoving volumes out to redshifts in an array. Flat LCDM Cosmology specified in global variables.


    Parameters:
        zs (array(float)): redshifts
        omega_matter (float): Matter density parameter at z=0

    Returns:
        array(float): Comoving volumes in Mpc^3 h^-3
    """
    vols = np.zeros(len(zs))
    for i in range(len(zs)):
        vols[i] = comoving_volume(zs[i], omega_matter)
    return vols 


@njit
def absolute_magnitude_limit(z, m_lim, omega_matter, h):
    """
    Convert apparent mag limit to absolute magnitude limit at redshift z.


    Parameters:
        z (float): redshift
        m_lim (float): Apparent magnitude limit of a given survey. 
        omega_matter (float): Matter density parameter at z=0
        h (float): Dimensionless Hubble parameter (H0 / 100 km/s/Mpc)

    Returns:
        float: Absolute magnitude limit at a given z
    """
    d_l = luminosity_distance(z, omega_matter) / h # in Mpc
    return m_lim - 5 * np.log10(d_l) - 25.0


@njit
def angular_sep(ra1, dec1, ra2, dec2):
    """
    Get the angular seperation between 2 RA/Dec coordinate pairs.


    Parameters:
        ra1 (float): Right ascension in degrees of 1st object
        dec1 (float): Declination in degrees of 1st object
        ra2 (float): Right ascension in degrees of 2nd object
        dec2 (float): Declination in degrees of 2nd object

    Returns:
        float: Angular seperation in degrees
    """
        
    d_ra = np.radians(ra2 - ra1)
    d_dec = np.radians(dec2 - dec1)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)

    a = np.sin(d_dec / 2.0)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2.0)**2
    c = 2. * np.arcsin(np.sqrt(a))
    return np.degrees(c)


@njit
def find_projected_separation(ra1, dec1, ra2, dec2, z1, omega_matter):
    """
    Get the projected seperation between 2 RA/Dec coordinate pairs at a given z. 


    Parameters:
        ra1 : float
            Right ascension in degrees of 1st object
        dec1 : float
            Declination in degrees of 1st object
        ra2 : float
            Right ascension in degrees of 2nd object
        dec2 : float
            Declination in degrees of 2nd object
        z1 : float
            Redshift of the 1st object

    Returns:
        float: Projected seperation in Mpc h ^-1
    """

    return np.radians(angular_sep(ra1, dec1, ra2, dec2))*comoving_distance(z1, omega_matter)  # in Mpc h^-1


@njit
def find_delta_z(z_group, z_gal):
    """
    Find the delta between 2 redshifts

    Parameters:
        z_group (float): Redshift of group (or 1st source)
        z_gal (float): Redshift of galaxy (or 2nd source)

    Returns:
        float: Redshift delta.
    """    
    return z_gal-z_group


@njit
def luminosity_to_mag(L, M_sun):
    """
    Convert luminosity to magnitude using the absolute magnitude of the Sun.

    Parameters:
        L (float): luminosity to convert to Magnitude. Expected in units of 10**14 * h-1 * L_sun
        M_sun (float): Absolute magnitude of the Sun in the z-band.

    Returns:
        float: magnitude in units of solar luminosities.
    """

    return -2.5 * np.log10(L * 10**14)+ M_sun  #Used to be no 2.5 and divided by 0.4 instead 


@njit
def get_all_luminosity_to_magnitude(L_array, M_sun):
    """
    Convert an array of luminosity to magnitude using the absolute magnitude of the Sun.

    Parameters:
        L_array (array(float)): luminosity to convert to Magnitude. Expected in units of 10**14 * h-1 * L_sun
        M_sun (float): Absolute magnitude of the Sun in the z-band.

    Returns:
        array(float): Magnitudes
    """
    mags = np.zeros(len(L_array))
    for i in range(len(L_array)):
        mags[i] = luminosity_to_mag(L_array[i], M_sun)
    return mags


@njit
def magnitude_to_luminosity(M, M_sun):
    """
    Convert magnitude to luminosity using the absolute magnitude of the Sun.

    Parameters:
        M (float): Absolute Magnitude to convert to luminosity.
        M_sun (float): Absolute magnitude of the Sun in the z-band.

    Returns:
        float: Luminosity in units of solar luminosities.
    """
    return 10 ** (0.4 * (M_sun - M))


@njit
def get_all_magnitude_to_luminosity(M_array, M_sun):
    """
    Convert an array of magnitude to luminosity using the absolute magnitude of the Sun.

    Parameters:
        M (array(float)): Absolute magnitude to convert to luminosity.
        M_sun (float): Absolute magnitude of the Sun in the z-band.

    Returns:
        array(float): Luminosity in units of 10**14 * h-1 * L_sun
    """
    luminosities = np.zeros(len(M_array))
    for i in range(len(M_array)):
        luminosities[i] = magnitude_to_luminosity(M_array[i], M_sun) / (1e14)
    return luminosities


@njit
def spherical_to_cartesian(ra, dec, comoving_distance):
    """
    Convert a single point from spherical coordinates to 3D Cartesian coordinates.
    
    Parameters:
    ----------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    comoving_distance : float
        Comoving distance
        
    Returns:
    -------
    numpy.ndarray
        1D array with x, y, z coordinates
    """
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    
    x = comoving_distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = comoving_distance * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = comoving_distance * np.sin(dec_rad)
    
    # Return a 1D array with 3 elements for a single point
    result = np.zeros(3)
    result[0] = x
    result[1] = y
    result[2] = z_coord
    return result


@njit
def find_all_spherical_to_cartesian(ra, dec, comoving_distance):
    """
    Convert arrays of spherical coordinates to 3D Cartesian coordinates.
    
    Parameters:
    ----------
    ra : array-like
        Right ascension in degrees
    dec : array-like
        Declination in degrees
    comoving_distance : array-like
        Comoving distances
        
    Returns:
    -------
    numpy.ndarray
        Array of shape (n, 3) with x, y, z coordinates
    """
    n = len(ra)
    all_coords = np.zeros((n, 3))
    
    for i in range(n):
        all_coords[i, 0] = comoving_distance[i] * np.cos(np.deg2rad(dec[i])) * np.cos(np.deg2rad(ra[i]))
        all_coords[i, 1] = comoving_distance[i] * np.cos(np.deg2rad(dec[i])) * np.sin(np.deg2rad(ra[i]))
        all_coords[i, 2] = comoving_distance[i] * np.sin(np.deg2rad(dec[i]))
    
    return all_coords
