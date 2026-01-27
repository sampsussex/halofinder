import numpy as np
from numba import njit
from cosmo_funcs import find_delta_z, find_projected_separation, Hubble
from astropy.constants import c, G
from astropy import units as u
#H = 0.704 # # Hubble constant in units of 100 km/s/Mpc
#H0 = 100 * H # Hubble constant in km/s/Mpc
#OM_M = 0.27 # Matter density parameter
#OM_L = 0.73 # Dark energy density parameter
#C = c.value/ 10**3 # in km/s
#change astropy g to # in (km/s)^2 Mpc/M_sun
299792.458

#G = G.to((u.km/u.s)**2 * u.Mpc / u.M_sun).value  # in (km/s)^2 Mpc/M_sun 4.300917270036279e-09



@njit
def find_rho_crit(z, omega_matter):
    """ Calculate the critical density at redshift z. Units are in M_sun / Mpc^3.
    Parameters:
    ----------
    z : float
        Redshift
    omega_matter : float
        Matter density parameter at z=0
    Returns:
    -------
    float
        Critical density at redshift z in M_sun / Mpc^3
    """
    G  = 4.300917270036279e-09 # G constant in (km/s)^2 Mpc/M_sun
    return 3. * Hubble(z, omega_matter)**2 / (8. * np.pi * G)

@njit
def find_Om(z, omega_matter):
    """ Calculate the matter density parameter at redshift z."""
    omega_lambda = 1. - omega_matter
    return omega_matter * (1 + z)**3 / (omega_matter * (1 + z)**3 + omega_lambda)

@njit
def find_halo_r(halo_mass, z_group, omega_matter, delta_crit=200.):
    """ Calculate the radius of a halo given its mass and redshift.
    Parameters:
    ----------
    halo_mass : float
        Halo mass in units of 10**14 h-1 M_sun
    z_group : float
        Redshift of the group
    omega_matter : float
        Matter density parameter at z=0
    delta_crit : float, optional
        Critical density to use. Default is 200.
    Returns:
    -------
    float
        Radius of the halo in Mpc h -1
    """

    mass = halo_mass * 1e14
    return (3 * mass / (4 * np.pi * delta_crit * find_rho_crit(z_group, omega_matter) * find_Om(z_group, omega_matter)))**(1/3)


@njit
def find_concentration_ratio(halo_mass, z = None, concentraion_relation = 'DuttonMaccio14', delta_crit= 200.):
    """ Calculate the concentration ratio of a halo given its mass and the concentration relation.
    Parameters:
    ----------
    halo_mass : float
        Halo mass in units of 10**14 h-1 M_sun
    concentraion_relation : str, optional
        Concentration relation to use. Default is 'DuttonMaccio14'.
    delta_crit : float, optional
        Critical density to use. Default is 200.
    Returns:
    -------
    float
        Concentration ratio of the halo. Units are dimensionless.
    Raises:
    ------
    ValueError
        If the concentration relation is not valid or if delta_crit is not 200 for Maccio relations.
    """
    if concentraion_relation == 'Maccio08':
        # Maccio 2008 concentration relation
        # Given as M200 for halo mass in the paper. Other options available in paper.
        if delta_crit != 200.:
            raise ValueError("delta_crit must be 200 for Maccio08 concentration ratio.")
        return 10**(0.830 - 0.098 * (np.log10(halo_mass*10**14)-12))

    
    if concentraion_relation == 'DuttonMaccio14':
        # Dutton Maccio 24 relation, scales with z. https://arxiv.org/pdf/1402.7073
        # What is going on with H here?
        if delta_crit != 200.: 
            raise ValueError("delta_cirt must be 200 for this implementation")
        a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * (z ** 1.21))
        b = b = -0.101 + 0.026 * z
        m = np.log10(halo_mass*1e14/1e12)
        return 10**(a + b * m)

    
    else:
        raise ValueError("Invalid concentration ratio method. Choose 'Maccio07' or 'Maccio08'.")


@njit
def find_scale_radius(halo_r, halo_concentration):
    """
    Returns the scale radius given r180 and the mass of the halo.
    Parameters:
    ----------
    halo_r : float
        Halo radius in units of Mpc
    halo_concentration : float
        Halo concentration ratio (dimensionless)
    Returns:
    -------
    float
        Scale radius of halo in units of Mpc

    """
    return  halo_r/halo_concentration


@njit
def f_x(scale_radius, projected_sep):
    """ Returns the NFW function f(x) for a given scale radius and projected separation."""
    x = projected_sep/scale_radius
    if abs(x - 1.0) < 1e-12: # Avoid numerical issues of sqrt near 0.
        return 1/3
    elif x < 1:
        return (1- np.log((1+np.sqrt(1-x**2)) / x) / np.sqrt(1-x**2)) / (x**2-1)
    elif x > 1:
        return (1- np.arctan(np.sqrt(x**2-1))/np.sqrt(x**2-1)) / (x**2-1)


@njit
def find_delta_bar(halo_concentration, delta_crit=200.):
    """ Calculate delta_bar for NFW profile.
    Parameters:
    ----------
    halo_concentration : float
        Halo concentration ratio (dimensionless)
    delta_crit : float, optional
        Critical density to use. Default is 200.
    Returns:
        -------
    float
        Delta bar for NFW profile. Units are dimensionless.
    """
    return delta_crit*halo_concentration**3/ (3*np.log(1+halo_concentration)- halo_concentration/(1+halo_concentration))


@njit
def find_NFW_sigma(projected_sep, halo_mass, z_group, omega_matter):
    """ Find the NFW sigma given a projected separation, halo mass and redshift of the group.
    Parameters:
    ----------
    projected_sep : float
        Projected separation in Mpc
    halo_mass : float
        Halo mass in units of 10**14 h^-1 M_sun
    z_group : float
        Redshift of the group
    omega_matter : float
        Matter density parameter at z=0
    Returns:
    -------
    float
        NFW sigma in Mpc
        
    """
    #Find NFW sigma
    halo_c = find_concentration_ratio(halo_mass, z_group)

    halo_r = find_halo_r(halo_mass, z_group, omega_matter)

    scale_radius = find_scale_radius(halo_r, halo_c)

    # If projected_sep is 0, return infinity
    if projected_sep == 0:
        return np.inf

    return 2*scale_radius*find_delta_bar(halo_c)*f_x(scale_radius, projected_sep)


# ----------------------
# Redshift membership
# ----------------------


@njit
def find_sigma_sqr(halo_mass, z_group, omega_matter, delta_crit = 200.):
    """ Find the velocity dispersion of a halo given its mass. 
    Parameters:
    ----------
    halo_mass : float
        Halo mass in units of 10**14 h-1 M_sun
    z_group : float
        Redshift of the group
    omega_matter : float
        Matter density parameter at z=0
    delta_crit : float, optional
        Critical density to use. Default is 200.
    Returns:
    -------
    float
        Velocity dispersion of the halo in km/s 
    """
    # return 632*(halo_mass*OM_M)**0.3224- old version from Yang+2021
    # Included the correct 3/5 prefactor from viral theorm
    G = 4.300917270036279e-09 # G constant in (km/s)^2 Mpc/M_sun
    return ((6**(2/3))/5)*np.pi**(1/3) * (halo_mass*1e14)**(2/3) * G * (1+z_group) * (delta_crit * find_rho_crit(z_group, omega_matter) * find_Om(z_group, omega_matter))**(1/3)


@njit
def find_p_delta_z(delta_z, z_group, halo_mass, omega_matter):
    """ Find the probability of a galaxy being at a given redshift given the group redshift and halo mass.
    Parameters:
    ----------
    delta_z : float
        Redshift difference between galaxy and group
    z_group : float
        Redshift of the group
    halo_mass : float
        Halo mass in units of 10**14 h-1 M_sun
    omega_matter : float
        Matter density parameter at z=0
    Returns:
    -------
    float
        Probability of a galaxy being at a given redshift. Units are dimensionless.
    """
    C = 299792.458 # Speed of light in km/s

    sigma_sqr = find_sigma_sqr(halo_mass, z_group, omega_matter)

    return C/(np.sqrt(2*np.pi)*np.sqrt(sigma_sqr)) * np.exp(-(C*delta_z)**2/(2*sigma_sqr))


# ------------------------
# Combined Membership criteria
# ------------------------


@njit
def find_p_M(ra1, dec1, ra2, dec2, z_group, z_gal, group_halo_mass, omega_matter, h):
    """
    Finds P_M(theta, z) in the style of yang 2007. This is the probability of a galaxies membership of a given group
    
    Parameters:
    ----------
    ra1 : float
        Right ascension in degrees of group ## Need to check it is this way round.
    dec1 : float
        Declination in degrees of group
    ra2 : float
        Right ascension in degrees of candidate galaxy
    dec2 : float
        Declination in degrees of candidate galaxy
    z_group : float
        Redshift of group
    z_gal : float
        Redshift of galaxy
    group_halo_mass : float
        Halo mass of the group in h^-1 / (M_sun*10**14)
        
    Returns:
    -------
    float
        Float with probabilty of membership. Units are dimensionless.
    """
    #if ra1 == ra2 and dec1 == dec2 and z_group == z_gal:
    #    return np.inf
    
    #else:
    projected_sep = find_projected_separation(ra1, dec1, ra2, dec2, z_group, omega_matter)

    delta_z = find_delta_z(z_gal, z_group)

    C = 299792.458

    H0 = h * 100.
    return H0/C * find_NFW_sigma(projected_sep, group_halo_mass, z_group, omega_matter) * find_p_delta_z(delta_z, z_group, group_halo_mass, omega_matter)
