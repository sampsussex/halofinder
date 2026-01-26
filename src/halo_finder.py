# Yang 07/21 groupfinder
import numpy as np
from astropy.table import Table
from numba_kdtree import KDTree
import logging
import matplotlib.pyplot as plt
from cosmo_funcs import get_all_comoving_distance, find_all_spherical_to_cartesian, get_all_magnitude_to_luminosity, get_all_luminosity_to_magnitude
from group_properties_funcs import find_all_initial_mass_to_light, brightest_galaxy_centers
from luminosity_funcs import update_halo_masses, k_corr, generate_hmf
from group_finding_funcs import update_group_membership_tinker
from utils import ConfigReader
from bijective_matching import s_score

#
# To do list - For MVP
# K corrections on group luminosities will be a nightmare but needs to be done
# 5logh needs to be in magnitudes, which are all in absolute mags
# Take in k corrections from input gal catalog.

# Score to beat: S_tot = 0.3601092310748732
#
# For paper
# Implement completeness correction
# Implement correction for survey edges (could this be related to above?)
# Think of novel way of adding photo-zs
# Red vs blue?

# Tidy ups; 

# 2) sort issues when group queries member galaxy
# 3) check that there are no halos without a central galaxy
# 4) Probably need to relook at logic on groupfindering to see if there are simpliciations
# 5) Flagging changes could be updating group properties a lot quicker. (Halo Mass probs not though)
# 7) rejig lf function. 

# ------------------------
# Set up logging
# ------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ------------------------
# YangFinder Class
# ------------------------
class HaloFinder:
    def __init__(self, config_reader):
        """
        Initialize HaloFinder with configuration from ConfigReader.
        
        Args:
            config_reader (ConfigReader): An instance of ConfigReader with loaded configuration
        """
        logging.info("Initializing HaloFinder...")
        
        # Initialize iteration tracking
        self.iteration_counter = 0
        self.iteration_group_membership = {}
        self.iteration_central_assignment = {}
        self.iteration_satellite_assignment = {}
        #self.iteration_central_is_red = {}

        # Get cosmology options from config
        cosmology = config_reader.get_cosmology()
        self.h = cosmology['h']

        self.omega_matter = cosmology['omega_matter']

        # Get survey fraction area from config
        self.survey_fractional_area = config_reader.get_survey_fractional_area()
        
        # Get setup options from config
        setup_options = config_reader.get_setup_options()
        self.max_iterations = setup_options['max_iterations']
        self.mag_limit = setup_options['survey_magnitude_limit']
        self.abs_mag_sun = setup_options['abs_solar_magnitude_in_band']
        self.remove_isolated_galaxies = setup_options['remove_isolated_galaxies']
        self.red_a_threshold = setup_options['red_a_threshold']
        self.red_b_threshold = setup_options['red_b_threshold']
        self.red_c_threshold = setup_options['red_c_threshold']
        self.blue_a_threshold = setup_options['blue_a_threshold']
        self.blue_b_threshold = setup_options['blue_b_threshold']
        self.blue_c_threshold = setup_options['blue_c_threshold']
        self.red_effective_luminosity_boost_a = setup_options['red_effective_luminosity_boost_a']
        self.red_effective_luminosity_boost_b = setup_options['red_effective_luminosity_boost_b']
        
        # Get file paths from config
        file_locations = config_reader.get_file_locations()
        self.data_load_path = file_locations['galaxy_catalog_path']
        self.save_path = file_locations['galaxy_group_path']
        self.lf_param_load_path = file_locations['luminosity_function_path']
        self.plot_save_dir = file_locations['plots_dir']
        self.s_tot_save_path = file_locations['s_tot_path']

        # get hmf setup options
        hmf_options = config_reader.get_hmf_options()
        self.hmf_min_mass = hmf_options['hmf_min_mass']
        self.hmf_max_mass = hmf_options['hmf_max_mass']
        self.hmf_redshift = hmf_options['hmf_redshift']
        self.hmf_dlog10m = hmf_options['hmf_dlog10m']

        # get lf params
        lf_options = config_reader.get_lf_options()
        self.lf_phi_star = lf_options['phi_star']
        self.lf_M_star = lf_options['M_star']
        self.lf_alpha = lf_options['alpha']

        # Get mock comparison options
        mock_options = config_reader.get_mock_comparison_options()
        self.bijective_matching_group_n_threshold = mock_options['group_n_threshold']


        # Store the config reader for potential future use
        self.config_reader = config_reader
        
        # Log initialization details
        logging.info(f"Max iterations set to {self.max_iterations}")
        logging.info(f"Magnitude limit set to {self.mag_limit}")
        logging.info(f"Data load path: {self.data_load_path}")
        logging.info(f"B parameter threshold: {self.b_threshold}")
        logging.info(f"Save path: {self.save_path}")


    def load_catalogue_data(self):
        """
        Load catalogue data using configuration from ConfigReader.
        Supports different data formats based on column mappings in config.
        """
        logging.info(f"Loading catalogue data from: {self.data_load_path}")
        
        # Load the data
        data = Table.read(self.data_load_path)
        logging.info(f"Data loaded with {len(data)} rows")
        
        # Get column name mappings from config
        column_names = self.config_reader.get_column_names()
        
        # Extract data using configured column names
        try:
            # Required columns
            self.gal_ids = np.array(data[column_names['galaxy_id']], dtype='int32')
            self.zobs = np.array(data[column_names['redshift']], dtype='float64')
            self.ra = np.array(data[column_names['ra']], dtype='float64')
            self.dec = np.array(data[column_names['dec']], dtype='float64')
            self.abs_mag = np.array(data[column_names['absolute_magnitude']], dtype='float64')
            self.k_corr = np.array(data[column_names['k_correction']], dtype='float64')
            self.is_red = np.array(data[column_names['galaxy_is_red']], dtype=bool)

            
            # Handle group ID - might be same as galaxy ID for some datasets
            if 'group_id' in column_names and column_names['group_id'] in data.colnames:
                self.id_group_sky = np.array(data[column_names['group_id']], dtype='int32')
            else:
                # If no group ID column, use galaxy ID as fallback
                logging.warning("No group ID column found")


            if self.remove_isolated_galaxies == True:
                logging.info("Removing isolated galaxies with group ID -1")
                mask = self.id_group_sky != -1
                self.gal_ids = self.gal_ids[mask]
                self.zobs = self.zobs[mask]
                self.ra = self.ra[mask]
                self.dec = self.dec[mask]
                self.abs_mag = self.abs_mag[mask]
                self.id_group_sky = self.id_group_sky[mask]
                self.k_corr = self.k_corr[mask]
                self.is_red = self.is_red[mask]
                logging.info(f"Isolated galaxies removed. Remaining galaxies: {len(self.gal_ids)}")
            
                
        except KeyError as e:
            logging.error(f"Column not found in data: {e}")
            logging.error(f"Available columns: {data.colnames}")
            raise KeyError(f"Required column {e} not found in catalogue data")
        
        # Clean up memory
        del data
        
        # Log what was loaded
        logging.info(f"Galaxy data extracted: {len(self.gal_ids)} galaxies")
        logging.info(f"Columns loaded: IDs, redshift, RA, Dec, absolute magnitude")


    def generate_hmf(self):
        logging.info("Generating Halo Mass Function (HMF)...")
        self.hmf_masses, self.hmf_mass_intervals = generate_hmf(self.hmf_redshift, self.hmf_min_mass, 
                                                                self.hmf_max_mass, self.hmf_dlog10m, 
                                                                self.h, self.omega_matter)
        logging.info(f"HMF generated with {len(self.hmf_masses)} mass bins from {self.hmf_masses[0]:.2e} to {self.hmf_masses[-1]:.2e} Msun/h")


    def get_all_comoving_distances(self):
        logging.info("Generating comoving distances for each galaxy...")
        self.gal_DMs = get_all_comoving_distance(self.zobs, self.omega_matter)
        plt.hist(self.gal_DMs)
        plt.title('Galaxy Distance moduli')
        plt.ylabel('Freq.')
        plt.xlabel('Comoving Distance, Mpc')
        plt.savefig('plots/galaxy_distance_moduli')
        logging.info("Comoving distances generated.")
        plt.clf()


    def create_KDE_tree(self):
        logging.info("Creating KDE of catalouge...")
        coords = find_all_spherical_to_cartesian(self.ra, self.dec, self.gal_DMs)
        self.gal_kde_tree = KDTree(coords)
        logging.info("KDE Tree successfully created")


    def initial_luminosities(self):
        logging.info("Generating initial group luminosities from Z band absolute magnitude...")
        self.gal_luminosities = get_all_magnitude_to_luminosity(self.abs_mag, self.abs_mag_sun)


        plt.hist(np.log10(self.gal_luminosities), log=True)
        plt.title('Initial galaxy luminosities')
        plt.ylabel('Freq.')
        plt.xlabel('log10(Luminosity / $10^{14}h^{-1}$)')
        plt.savefig('plots/initial_galaxy_luminosities')
        plt.clf()

        logging.info("Luminosities found.")


    def initial_group_central_satellite_assignment(self):
        """
        Assign all galaxies as centrals in the first iteration.
        """
        logging.info("Assigning initial central and satellite galaxies...")
        self.gal_is_central = np.ones(len(self.gal_ids), dtype=bool)  # All galaxies are considered centrals initially
        self.gal_is_satellite = np.zeros(len(self.gal_ids), dtype=bool)  # No satellites initially
        self.group_ids = np.arange(len(self.gal_ids), dtype='int32')  # Each galaxy is its own group initially
        self.iteration_group_membership[str(self.iteration_counter)] = self.group_ids.copy()
        self.iteration_central_assignment[str(self.iteration_counter)] = self.gal_is_central.copy()
        self.iteration_satellite_assignment[str(self.iteration_counter)] = self.gal_is_satellite.copy()
        logging.info("Initial central and satellite assignment complete.")


    def update_group_luminosity_and_centres(self):
        logging.info("Updating unique group list, luminosity weighted group centres and luminosities...")
        
        #self.unique_groups, self.group_centres_ra, self.group_centres_dec, self.group_centres_z, self.group_luminosities, self.group_sizes= luminosity_weighted_centers(self.gal_luminosities, self.ra, self.dec, self.zobs, self.group_ids, self.phi_star, self.M_star, self.alpha, self.mag_limit) 
        self.unique_groups, self.group_centres_ra, self.group_centres_dec, self.group_centres_z, self.group_luminosities, self.group_bcg_abs_mag, self.group_sizes = brightest_galaxy_centers(self.gal_luminosities, self.abs_mag, self.ra, 
                                                                                                                                                                      self.dec, self.zobs, self.group_ids, 
                                                                                                                                                                      self.lf_phi_star, self.lf_M_star, self.lf_alpha, 
                                                                                                                                                                      self.mag_limit, self.omega_matter, 
                                                                                                                                                                      self.h) 
        logging.info('Lists updated successfully')

    def update_group_halo_masses(self):
        logging.info("Updating halo masses from Luminosity rank assignment...")
        self.group_bcg_k_corrs = k_corr(self.group_centres_z)
        

        self.group_magnitudes = get_all_luminosity_to_magnitude(self.group_luminosities, self.abs_mag_sun)

        plt.hist(np.log10(self.group_luminosities*1e14), log=True, bins=25)
        plt.title('Halo Luminosity Histogram Pre Mass Assignment')
        plt.savefig(f'{self.plot_save_dir}/halo_luminosities_iter_{self.iteration_counter}_pre_mass_assignment.png')
        plt.clf()
        plt.hist(self.group_bcg_k_corrs)
        plt.title('Group K-corrections')
        plt.savefig(f'{self.plot_save_dir}/group_k_corrections_iter_{self.iteration_counter}.png')
        plt.clf()

        plt.hist(self.group_magnitudes, log=True, bins = 25)
        plt.title('Halo Magnitude Histogram Pre Mass Assignment')
        plt.xlabel('Absolute Magnitude')
        plt.savefig(f'{self.plot_save_dir}/halo_magnitudes_iter_{self.iteration_counter}_pre_mass_assignment.png')
        plt.clf()

        plt.hist(self.group_centres_z)
        plt.title('Group Redshift Histogram')
        plt.xlabel('Redshift')
        plt.savefig(f'{self.plot_save_dir}/group_redshifts_iter_{self.iteration_counter}.png')
        plt.clf()
        logging.info(f"mag limits: {self.mag_limit}, survey fractional area: {self.survey_fractional_area}")


        plt.bar(np.log10(self.hmf_masses), self.hmf_mass_intervals, width=0.1)
        plt.title('Halo Mass Function Intervals')
        plt.xlabel('log10(Halo Mass / $10^{14}h^{-1}$)')
        plt.ylabel('dn/dlogM')
        plt.savefig(f'{self.plot_save_dir}/hmf_intervals_iter_{self.iteration_counter}.png')
        plt.clf()



        self.group_halo_masses = update_halo_masses(self.group_magnitudes, self.group_centres_z,
                                                    self.group_bcg_abs_mag, self.group_bcg_k_corrs, self.mag_limit, 
                                                    self.survey_fractional_area, self.hmf_masses, 
                                                    self.hmf_mass_intervals, self.omega_matter, 
                                                    self.h)
        plt.hist(np.log10(1e14*self.group_halo_masses), log=True, bins=25)

        plt.savefig(f'{self.plot_save_dir}/halo_masses_iter_{self.iteration_counter}_post_mass_assignment.png')
        plt.clf()

        logging.info("Halo masses updated.")

    def apply_tinker_finder(self):
        logging.info("Performing iteration of group finder")
        self.new_members, self.gal_is_central, self.gal_is_satellite = update_group_membership_tinker(
            self.ra, self.dec, self.zobs, self.group_ids,
            self.unique_groups, self.group_centres_ra, self.group_centres_dec, self.group_centres_z, 
            self.group_sizes, self.group_halo_masses, self.gal_kde_tree, self.gal_is_central, self.gal_is_satellite, 
            self.is_red, self.red_a_threshold, self.red_b_threshold, self.red_c_threshold, self.blue_a_threshold, 
            self.blue_b_threshold, self.blue_c_threshold, self.omega_matter, self.h)
        logging.info(f"Groupfinder iteration complete, number of groups found: {len(np.unique(self.new_members))}")


    def save_galaxy_groups(self):
        logging.info(f"Saving galaxy groups at the location: {self.save_path}")
        header = "id_galaxy_sky\tid_finder_group"
        data = np.column_stack((self.gal_ids, self.group_ids))
        np.savetxt(self.save_path, data, header=header)


    def iterate_tinker_finder(self):
        while self.iteration_counter < self.max_iterations:
            self.iteration_counter +=1
            logging.info(f"Starting iteration number : {self.iteration_counter}...")
            self.apply_tinker_finder()
            changed_group_values = np.sum(self.new_members != self.group_ids)
            logging.info(f"{changed_group_values} galaxies have changed group membership last iteration")
            if changed_group_values == 0:
                logging.info(f"Ending finder iteration, no changes in membership detected")
                break
            else:
                self.iteration_group_membership[str(self.iteration_counter)]  = self.new_members.copy()
                self.group_ids = self.new_members.copy()
                self.update_group_luminosity_and_centres()
                self.update_group_halo_masses()
                self.debugging_plots()
                logging.info(f"Iteration number : {self.iteration_counter} complete")
                if self.iteration_counter >= self.max_iterations:
                    logging.info("Max number of iterations complete, ending finder.")
        self.save_galaxy_groups()


    def s_score(self):
        """
        Calculate the s-score for the current group assignment.
        The s-score is a measure of the quality of the group assignment.
        """
        logging.info("Calculating s-score for current group assignment...")
        
        # Calculate S-score
        score, e_tot, q_tot= s_score(self.group_ids, self.id_group_sky, 5)
        logging.info(f"S-score calculated: {score:.6f}, e_tot: {e_tot:.6f}, q_tot: {q_tot:.6f}")

        self.s_tot = score
        # Save the S-score to a file
        np.savetxt(self.s_tot_save_path + '_B:' + str(self.b_threshold), [score], header='S-score', fmt='%.6f')


    def debugging_plots(self):
        logging.info("Creating debugging plots...")
        #Halo masses
        plt.hist(np.log10(self.group_halo_masses * 1e14), log=True, bins=25)
        plt.title('Halo Mass Histogram')
        plt.xlabel('log10(Halo Mass / $h^{-1}$)')
        plt.ylabel('Freq.')
        plt.savefig(f'{self.plot_save_dir}/halo_masses_iter_{self.iteration_counter}.png')
        plt.clf()

        #Halo luminosities
        plt.hist(np.log10(self.group_luminosities*1e14), log=True, bins = 25)
        plt.title('Halo Luminosity Histogram')
        plt.xlabel('log10(Luminosity)')
        plt.ylabel('Freq.')
        plt.savefig(f'{self.plot_save_dir}/halo_luminosities_iter_{self.iteration_counter}.png')
        plt.clf()

        #Halo N
        plt.hist(self.group_sizes, log=True, bins = 50)
        plt.title('Halo Population counts')
        plt.ylabel("Freq.")
        plt.xlabel('Galaxy number count per halo')
        plt.savefig(f'{self.plot_save_dir}/halo_population_counts_iter_{self.iteration_counter}.png')
        plt.clf()

        #Halo locations
        plt.scatter(self.group_centres_ra, self.group_centres_dec, s = 0.1)
        plt.title('Halo locations')
        plt.xlabel('RA')
        plt.ylabel('Dec')

        plt.savefig(f'{self.plot_save_dir}/halo_locations_iter_{self.iteration_counter}.png')
        plt.clf()


        # Halo L vs M
        plt.scatter(np.log10(self.group_halo_masses*1e14), np.log10(self.group_luminosities*1e14), s=0.1)
        plt.title("Halo mass vs Luminosity")
        plt.xlabel('log10(Halo Mass / h^{-1}$)')
        plt.ylabel('log10(Luminosity /h^{-1}$)')

        plt.savefig(f'{self.plot_save_dir}/halo_m_vs_l_iter_{self.iteration_counter}.png')
        plt.clf()
        logging.info("Debugging plots created.")
        #Going to plot basically everything here to see if theres a hickup



        

# ------------------------
# Example Usage
# ------------------------

class TinkerFinder(HaloFinder):
    def __init__(self, config_reader):
        """
        Initialize TinkerFinder with configuration from ConfigReader.
        
        Args:
            config_reader (ConfigReader): An instance of ConfigReader with loaded configuration
        """
        super().__init__(config_reader)
        logging.info("TinkerFinder initialized successfully")
    
    def run(self):
        logging.info("Running the Tinker Finder...")
        self.load_catalogue_data()
        self.generate_hmf()
        self.get_all_comoving_distances()
        self.create_KDE_tree()
        self.initial_group_central_satellite_assignment()
        self.initial_luminosities()
        self.update_group_luminosity_and_centres()
        self.update_group_halo_masses()
        self.iterate_tinker_finder()
        self.s_score()
        logging.info("Tinker Finder run complete.")


#if __name__ == "__main__":
#    # Load configuration
#    config_reader = ConfigReader("config_galform.yaml")
#    config_reader.load_config()
#    #config_reader.validate_config()
#    # Create and run TinkerFinder
#    
#    finder = TinkerFinder(config_reader)
#    finder.run()
    
#    logging.info("Finder run complete")
