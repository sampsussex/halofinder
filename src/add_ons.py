from nessie import RedshiftCatalog, FlatCosmology
from  nessie.helper_funcs  import  create_density_function
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import matplotlib.pyplot as plt


def mpc_to_angle_deg(z, size=1.0*u.Mpc, omega_matter=0.3, h=0.6751):
    cosmo = FlatLambdaCDM(H0=h*100, Om0=omega_matter)
    z = np.asarray(z)
    theta = size / cosmo.angular_diameter_distance(z)  # radians
    return theta.to(u.deg, equivalencies=u.dimensionless_angles()).value


def get_completeness(ra_targeted, dec_targeted, ra_observed, dec_observed, redshift_observed):
    cosmo = FlatCosmology(h = 0.7, omega_matter = 0.3)
    running_density = create_density_function(redshift_observed, total_counts = len(redshift_observed), 
                                            survey_fractional_area = 0.1, cosmology = cosmo)

    nessie_cat = RedshiftCatalog(ra_observed, dec_observed, redshift_observed, running_density, cosmo)
    # need to find angle that defines 1mpc away from each galaxy at its redshift. 
    angles = mpc_to_angle_deg(redshift_observed, size=1.0*u.Mpc)
    nessie_cat.calculate_completeness(ra_targeted, dec_targeted, angles)
    return nessie_cat.completeness


def reformat_sharks(in_filepath, out_filepath, peturbed_stellar_masses=False, region= 'wide', masked=False):
    if region not in ['wide', 'deep']:
        raise ValueError("Region must be 'wide' or 'deep'")
    
    if masked not in [True, False]:
        raise ValueError("Mask must be True or False")

    h = 0.6751

    # columns to read
    cols = ['ra', 'dec', 
            'id_galaxy_sky', 
            'redshift_observed', 
            'mass_stellar_total', 
            'mass_stellar_disk', 
            'mass_stellar_bulge', 
            'mass_virial_hosthalo', 
            'mass_virial_subhalo', 
            'sfr_total', 
            'id_group_sky', 
            'mag_r_SDSS',
            'id_fof', 
            'masked', 
            'mag_r_VST', 
            'mag_Z_VISTA', 
            'mag_abs_Z_VISTA', 
            'mag_abs_r_VST',
            'mag_abs_r_SDSS', 
            'observed', 
            'ghosted']
    


    sharks = pd.read_parquet(in_filepath, columns=cols)
    sharks['mass_stellar_total'] = np.log10(sharks['mass_stellar_total'])

    sharks['log_sSFR'] = np.log10(sharks['sfr_total'] + 1e-10) - sharks['mass_stellar_total']


    print('checking hs in mass stellar total')
    # I know that mass_stellar_disk and mass_stellar_bulge is in Msun/h
    # So, for the first 5 rows, I will check that mass_stellar_total is equal to log10 mass_stellar_disk + mass_stellar_bulge, and that they are all in Msun/h
    #for i in range(5):
    #    total = sharks['mass_stellar_total'].iloc[i]
    #    disk = sharks['mass_stellar_disk'].iloc[i]
    #    bulge = sharks['mass_stellar_bulge'].iloc[i]
    #    print(f"Row {i}: total={total}, NO DIV hlog10(sum)={np.log10(disk+bulge)}, DIV h log10(sum)={np.log10((disk+bulge)/0.6751)}")
    # mass_stellar_total matches the div by h version.
    # as mass_stellar_disk is in Msun/h and mass_stellar_bulge is in Msun/h, when these
    # are divided by h, they are in Msun. So, to convert mass_stellar_total to Msun/h, I need to times by h.
    # 

    plt.hist(sharks['mass_stellar_total'])
    plt.show()
    plt.hist(np.log10(sharks['mass_stellar_total']))
    plt.show()

    mask = (sharks['mass_stellar_total'] > 8) & (sharks['mag_Z_VISTA'] > -99) #& (sharks['mag_Z_VISTA'] < 21.1) #& (sharks['dec'] > -3.95)
    if region == 'wide':
        mask = mask & (sharks['mag_Z_VISTA'] < 21.1) & (sharks['redshift_observed'] < 0.2)
    elif region == 'deep':
        mask = (
                mask & 
                (sharks['mag_Z_VISTA'] < 21.25) & 
                (sharks['dec'] > -35.0) &
                (sharks['dec'] < -30) &
                (sharks['ra'] > 339) &
                (sharks['ra'] < 351) & (sharks['redshift_observed'] < 0.8)
        )



    sharks = sharks[mask].reset_index(drop=True)

    #plt.hist(sharks['log_sSFR'], bins=50)
    #plt.xlabel('log sSFR')
    #plt.ylabel('Number of galaxies')
    #plt.show()

    # I need to caclulate the sfr and save the is_red flag
    sharks['is_red'] = sharks['log_sSFR'] < -11
    # I need to caculate the k corrections back from the abs and app mags.
    cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3)
    sharks['DM'] = cosmo.distmod(sharks['redshift_observed']).value

    #data['Rpetro_abs'] = data['Rpetro'] - data['DM'] - data['k-e corr']
    sharks['k-e corr'] = -sharks['mag_abs_Z_VISTA'] + sharks['mag_Z_VISTA'] - sharks['DM']

    # Plot histogram of k-e corr

    plt.hist(sharks['k-e corr'], bins=50)
    plt.xlabel('k-e corr (Z band)')
    plt.ylabel('Number of galaxies')
    plt.show()

    plt.hist(sharks['redshift_observed'], bins=50, log = True)
    plt.xlabel('Observed Redshift')
    plt.ylabel('Number of galaxies')
    #plt.xscale('symlog')
    plt.show()
    sharks['mass_stellar_total'] = sharks['mass_stellar_total'] + np.log10(h)
    sharks['stellar_mass'] = 10**(sharks['mass_stellar_total'])# * h
    #sharks['mass_stellar_total'] = sharks['mass_stellar_total'] + np.log10(h)

    plt.hist(sharks['stellar_mass'], bins=50, log=True)
    plt.xlabel('Stellar Mass (Msun)')
    plt.ylabel('Number of galaxies')
    plt.show()
    if peturbed_stellar_masses:
        print("Peturbing Stellar Masses with a Gaussian noise of 0.2 dex")
        sharks['log_stellar_mass'] = sharks['log_stellar_mass'] + np.random.normal(0, 0.2, size=len(sharks))
        sharks['stellar_mass'] = 10**sharks['log_stellar_mass']

    if masked == False:
        sharks['completeness'] = np.ones(len(sharks))
        sharks.to_parquet(out_filepath, index=False)

    elif masked == True:
        sharks_observed = sharks[(sharks['observed'] == True) & (sharks['ghosted']==False)].reset_index(drop=True)
        sharks_observed['completeness'] = get_completeness(sharks['ra'], sharks['dec'], sharks_observed['ra'], sharks_observed['dec'], sharks_observed['redshift_observed'])
        sharks_observed.to_parquet(out_filepath, index=False)
    

# I need to find and calculate the survey fractional sky area. 

def rectangular_sky_area_deg2(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg):
    """
    Area on the celestial sphere for an RA/Dec-aligned rectangle.

    Handles RA wrap-around (e.g. ra_min=350, ra_max=20).
    Inputs/outputs in degrees / deg^2.
    """
    ra_min = ra_min_deg % 360.0
    ra_max = ra_max_deg % 360.0

    # RA span in degrees, wrap-safe
    d_ra = (ra_max - ra_min) % 360.0  # in [0, 360)
    if np.isclose(d_ra, 0.0) and not np.isclose(ra_max_deg, ra_min_deg):
        d_ra = 360.0  # full wrap (rare edge case)

    # spherical rectangle area: Δλ * (sin δ2 - sin δ1)
    d_lambda = np.deg2rad(d_ra)
    sin_term = np.sin(np.deg2rad(dec_max_deg)) - np.sin(np.deg2rad(dec_min_deg))
    area_sr = d_lambda * sin_term

    # convert sr -> deg^2
    area_deg2 = area_sr * (180.0 / np.pi) ** 2
    print(f"Calculated area: {area_deg2:.2f} deg^2 for RA [{ra_min_deg}, {ra_max_deg}] and Dec [{dec_min_deg}, {dec_max_deg}]")
    return area_deg2


def rectangular_fraction_of_sky(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg):
    area_deg2 = rectangular_sky_area_deg2(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg)
    full_sky_deg2 = 4.0 * np.pi * (180.0 / np.pi) ** 2  # ~41252.96
    print(f"Calculated area: {area_deg2:.2f} deg^2, which is {area_deg2/full_sky_deg2:.6f} of the full sky.")
    return area_deg2 / full_sky_deg2



def main():
    infile_train = '/Users/sp624AA/Downloads/groupfinding_comp_mocks/fibre_incomplete_mocks.parquet'
    infile_comp = '/Users/sp624AA/Downloads/groupfinding_comp_mocks/shark_galaxies_comp.parquet'

    to_run = ['wide', 'deep']
    masked = True

    #out_train
    reformat_sharks(infile_train, '/Users/sp624AA/Downloads/groupfinding_comp_mocks/train_wide.parquet', region = 'wide', masked=False)
    reformat_sharks(infile_train, '/Users/sp624AA/Downloads/groupfinding_comp_mocks/train_deep.parquet', region = 'deep', masked=False)

    reformat_sharks(infile_train, '/Users/sp624AA/Downloads/groupfinding_comp_mocks/train_wide_masked.parquet', region = 'wide', masked=True)
    reformat_sharks(infile_train, '/Users/sp624AA/Downloads/groupfinding_comp_mocks/train_deep_masked.parquet', region = 'deep', masked=True)

    #reformat_sharks(infile_comp, '/Users/sp624AA/Downloads/groupfinding_comp_mocks/comp_wide.parquet', region = 'wide', masked=True)
    #reformat_sharks(infile_comp, '/Users/sp624AA/Downloads/groupfinding_comp_mocks/comp_deep.parquet', region = 'deep', masked=True)

if __name__ == "__main__":
    main()
