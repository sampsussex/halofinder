# Yang 07/21 groupfinder
import numpy as np
from astropy.table import Table
from numba_kdtree import KDTree
import logging
import matplotlib.pyplot as plt
from cosmo_funcs import (
    get_all_comoving_distance,
    find_all_spherical_to_cartesian,
    get_all_magnitude_to_luminosity,
    get_all_luminosity_to_magnitude,
)
from group_properties_funcs import (
    find_all_initial_mass_to_light,
    brightest_galaxy_centers,
    brightest_galaxy_centers_fast,
    calculate_group_dynamical_masses,
    fit_log_luminosity_log_mass_relation,
)
from luminosity_mass_funcs import (
    k_corr,
    generate_hmf,
    abundance_match_halo_masses,
    linear_stellar_mass2halo_mass,
    stellar2halo_mass_van_kampen,
    linear_luminosity2halo_mass,
    red_blue_linear_luminosity2halo_mass
)
from group_finding_funcs import update_group_membership_halofinder
from utils import ConfigReader
from bijective_matching import s_score

# To do list - For MVP
# 5logh needs to be in magnitudes, which are all in absolute mags
# //TODO fix high mass end of hmf (where i always get 1 of the highest possible mass in the hmf range)


# For paper
# Implement completeness correction
# Implement correction for survey edges (could this be related to above?)
# Think of novel way of adding photo-zs


# ------------------------
# Set up logging
# ------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ------------------------
# Halofinder Class
# ------------------------
class HaloFinder:
    _shared_cache = {}

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
        self.active_group_ids = None
        # self.iteration_central_is_red = {}

        # Get run options from config
        run_options = config_reader.get_run_options()
        self.make_plots = run_options.get("make_plots", True)
        self.use_shared_cache = run_options.get(
            "optimse_on_mock", False
        ) or run_options.get("optimse_parameter_space", False)

        self.mass_assignment_mode = run_options["mode"]
        self.run_mock_comparion = run_options["run_mock_comparison"]
        self.optimse_parameter_space = run_options["optimse_parameter_space"]
        self.optimse_on_mock = run_options["optimse_on_mock"]
        # Get cosmology options from config
        cosmology = config_reader.get_cosmology()
        self.h = cosmology["h"]

        self.omega_matter = cosmology["omega_matter"]

        # Get survey fraction area from config
        self.survey_fractional_area = config_reader.get_survey_fractional_area()

        # Get setup options from config
        finder_options = config_reader.get_finder_options()
        threshold_options = config_reader.get_threshold_model_params()
        shmr_params = config_reader.get_shmr_params()
        lhmr_params = config_reader.get_lhmr_params()
        red_blue_lhmr_params = config_reader.get_red_blue_lhmr_params()
        lhmr_dynamical_params = config_reader.get_lhmr_dynamical_calibrated_params()

        self.max_iterations = finder_options["max_iterations"]
        self.mag_limit = finder_options["survey_magnitude_limit"]
        self.abs_mag_sun = finder_options["abs_solar_magnitude_in_band"]
        self.remove_isolated_galaxies = finder_options["remove_isolated_galaxies"]

        self.red_a_threshold = threshold_options["red_a_threshold"]
        self.red_b_threshold = threshold_options["red_b_threshold"]
        self.blue_a_threshold = threshold_options["blue_a_threshold"]
        self.blue_b_threshold = threshold_options["blue_b_threshold"]
        self.threshold_b_pivot = threshold_options["threshold_b_pivot"]

        self.shmr_slope = shmr_params["shmr_slope"]
        self.shmr_intercept = shmr_params["shmr_intercept"]
        self.shmr_method = shmr_params.get("method", "linear")
        self.lhmr_slope = lhmr_params["lhmr_slope"]
        self.lhmr_intercept = lhmr_params["lhmr_intercept"]
        self.lhmr_slope_red = red_blue_lhmr_params["lhmr_slope_red"]
        self.lhmr_intercept_red = red_blue_lhmr_params["lhmr_intercept_red"]
        self.lhmr_slope_blue = red_blue_lhmr_params["lhmr_slope_blue"]
        self.lhmr_intercept_blue = red_blue_lhmr_params["lhmr_intercept_blue"]

        self.lhmr_dyn_A = lhmr_dynamical_params.get("A", 1.0)
        self.lhmr_dyn_min_group_members = lhmr_dynamical_params.get("min_group_members", 5)
        self.lhmr_dyn_current_slope = self.lhmr_slope
        self.lhmr_dyn_current_intercept = self.lhmr_intercept

        # self.b_threshold = setup_options.get('b_threshold', 0.0)

        # Get file paths from config
        file_locations = config_reader.get_file_locations()
        self.data_load_path = file_locations["galaxy_catalog_path"]
        self.save_path = file_locations["galaxy_group_path"]
        self.plot_save_dir = file_locations["plots_dir"]
        self.s_tot_save_path = file_locations["s_tot_path"]

        # get hmf setup options
        hmf_options = config_reader.get_hmf_options()
        self.hmf_min_mass = hmf_options["hmf_min_mass"]
        self.hmf_max_mass = hmf_options["hmf_max_mass"]
        self.hmf_redshift = hmf_options["hmf_redshift"]
        self.hmf_dlog10m = hmf_options["hmf_dlog10m"]

        # get lf params
        lf_options = config_reader.get_lf_options()
        self.lf_phi_star = lf_options["phi_star"]
        self.lf_M_star = lf_options["M_star"]
        self.lf_alpha = lf_options["alpha"]

        # Get mock comparison options
        mock_options = config_reader.get_mock_comparison_options()
        self.bijective_matching_group_n_threshold = mock_options["group_n_threshold"]

        # Store the config reader for potential future use
        self.config_reader = config_reader

        # Log initialization details
        logging.info(f"Max iterations set to {self.max_iterations}")
        logging.info(f"Mass assignemnt mode set to {self.mass_assignment_mode}")
        logging.info(f"Magnitude limit set to {self.mag_limit}")
        logging.info(f"Data load path: {self.data_load_path}")
        logging.info(f"Save path: {self.save_path}")
        logging.info(f"HaloFinder Initialised.")

    def _cache_key(self):
        return (
            self.data_load_path,
            self.hmf_min_mass,
            self.hmf_max_mass,
            self.hmf_redshift,
            self.hmf_dlog10m,
            self.omega_matter,
            self.h,
            self.remove_isolated_galaxies,
            self.mass_assignment_mode,
        )

    def _get_cache(self):
        if not self.use_shared_cache:
            return None
        return self._shared_cache.setdefault(self._cache_key(), {})

    def load_catalogue_data(self):
        """
        Load catalogue data using configuration from ConfigReader.
        Supports different data formats based on column mappings in config.
        """
        logging.info(f"Loading catalogue data from: {self.data_load_path}")

        cache = self._get_cache()
        if cache and "catalogue" in cache:
            cached = cache["catalogue"]
            self.gal_ids = cached["gal_ids"].copy()
            self.zobs = cached["zobs"].copy()
            self.ra = cached["ra"].copy()
            self.dec = cached["dec"].copy()
            self.abs_mag = cached["abs_mag"].copy()
            self.k_corr = cached["k_corr"].copy()
            self.is_red = cached["is_red"].copy()
            if self.mass_assignment_mode == "shmr":
                self.stellar_mass = cached["stellar_mass"].copy()
            self.id_group_sky = cached["id_group_sky"].copy()
            logging.info("Loaded catalogue data from cache.")
            return

        # Load the data
        data = Table.read(self.data_load_path)
        logging.info(f"Data loaded with {len(data)} rows")

        # Get column name mappings from config
        column_names = self.config_reader.get_column_names()

        # Extract data using configured column names
        try:
            # Required columns
            self.gal_ids = np.array(data[column_names["galaxy_id"]], dtype="int32")
            self.zobs = np.array(data[column_names["redshift"]], dtype="float64")
            self.ra = np.array(data[column_names["ra"]], dtype="float64")
            self.dec = np.array(data[column_names["dec"]], dtype="float64")
            self.abs_mag = np.array(
                data[column_names["absolute_magnitude"]], dtype="float64"
            )
            self.k_corr = np.array(data[column_names["k_correction"]], dtype="float64")
            self.is_red = np.array(data[column_names["galaxy_is_red"]], dtype=bool)
            if self.mass_assignment_mode == "shmr":
                self.stellar_mass = np.array(
                    data[column_names["stellar_mass"]], dtype="float64"
                )

            # Handle group ID - might be same as galaxy ID for some datasets
            if "group_id" in column_names and column_names["group_id"] in data.colnames:
                self.id_group_sky = np.array(
                    data[column_names["group_id"]], dtype="int32"
                )
            else:
                run_options = self.config_reader.get_run_options()
                if (
                    run_options.get("run_mock_comparison")
                    or run_options.get("optimse_on_mock")
                    or run_options.get("optimse_parameter_space")
                ):
                    raise ValueError(
                        "Mock comparison requires a group_id column in the input data."
                    )
                logging.warning("No group ID column found")

            if self.remove_isolated_galaxies:
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
                if self.mass_assignment_mode == "shmr":
                    self.stellar_mass = self.stellar_mass[mask]
                logging.info(
                    f"Isolated galaxies removed. Remaining galaxies: {len(self.gal_ids)}"
                )

        except KeyError as e:
            logging.error(f"Column not found in data: {e}")
            logging.error(f"Available columns: {data.colnames}")
            raise KeyError(f"Required column {e} not found in catalogue data")

        # Clean up memory
        del data

        # Log what was loaded
        logging.info(f"Galaxy data extracted: {len(self.gal_ids)} galaxies")

        if cache is not None:
            cache_catalogue = {
                "gal_ids": self.gal_ids.copy(),
                "zobs": self.zobs.copy(),
                "ra": self.ra.copy(),
                "dec": self.dec.copy(),
                "abs_mag": self.abs_mag.copy(),
                "k_corr": self.k_corr.copy(),
                "is_red": self.is_red.copy(),
                "id_group_sky": self.id_group_sky.copy(),
            }
            if self.mass_assignment_mode == "shmr":
                cache_catalogue["stellar_mass"] = self.stellar_mass.copy()
            cache["catalogue"] = cache_catalogue

    def generate_hmf(self):
        logging.info("Generating Halo Mass Function (HMF)...")
        cache = self._get_cache()
        if cache and "hmf" in cache:
            cached = cache["hmf"]
            self.hmf_masses = cached["hmf_masses"].copy()
            self.hmf_mass_intervals = cached["hmf_mass_intervals"].copy()
            logging.info("Loaded HMF from cache.")
            return
        self.hmf_masses, self.hmf_mass_intervals = generate_hmf(
            self.hmf_redshift,
            self.hmf_min_mass,
            self.hmf_max_mass,
            self.hmf_dlog10m,
            self.h,
            self.omega_matter,
        )
        logging.info(
            "HMF generated with %d mass bins from %.2e to %.2e Msun/h",
            len(self.hmf_masses),
            self.hmf_masses[0],
            self.hmf_masses[-1],
        )
        if cache is not None:
            cache["hmf"] = {
                "hmf_masses": self.hmf_masses.copy(),
                "hmf_mass_intervals": self.hmf_mass_intervals.copy(),
            }

    def get_all_comoving_distances(self):
        logging.info("Generating comoving distances for each galaxy...")
        cache = self._get_cache()
        if cache and "comoving_distances" in cache:
            self.gal_DMs = cache["comoving_distances"].copy()
            logging.info("Loaded comoving distances from cache.")
            return
        self.gal_DMs = get_all_comoving_distance(self.zobs, self.omega_matter)
        if self.make_plots:
            plt.hist(self.gal_DMs)
            plt.title("Galaxy Comoving Distance Distribution")
            plt.ylabel("Frequency")
            plt.xlabel("Comoving Distance [Mpc]")
            plt.savefig(f"{self.plot_save_dir}/galaxy_comoving_distance_distribution.png")
            plt.clf()
        logging.info("Comoving distances generated.")
        if cache is not None:
            cache["comoving_distances"] = self.gal_DMs.copy()

    def create_KDE_tree(self):
        logging.info("Creating KDE of catalouge...")
        cache = self._get_cache()
        if cache and "kde_tree" in cache:
            self.gal_kde_tree = cache["kde_tree"]
            logging.info("Loaded KDE Tree from cache.")
            return
        coords = find_all_spherical_to_cartesian(self.ra, self.dec, self.gal_DMs)
        self.gal_kde_tree = KDTree(coords)
        logging.info("KDE Tree successfully created")
        if cache is not None:
            cache["kde_tree"] = self.gal_kde_tree

    def initial_luminosities(self):
        logging.info(
            f"Generating initial group luminosities from absolute magnitude..."
        )
        self.gal_luminosities = get_all_magnitude_to_luminosity(
            self.abs_mag, self.abs_mag_sun
        )

        if self.make_plots:
            plt.hist(np.log10(self.gal_luminosities), log=True)
            plt.title("Initial Galaxy Luminosity Distribution")
            plt.ylabel("Frequency")
            plt.xlabel(r"log10(Galaxy Luminosity [$10^{14} h^{-2} L_{\odot}$])")
            plt.savefig(f"{self.plot_save_dir}/initial_galaxy_luminosities.png")
            plt.clf()

        logging.info("Luminosities generated.")

    def initial_mass_assignment(self):
        logging.info("Generating initial halo masses from mass-to-light ratio...")
        self.group_halo_masses = find_all_initial_mass_to_light(
            self.group_luminosities, 100.0
        )
        logging.info("Initial halo masses assigned.")

    def initial_group_central_satellite_assignment(self):
        """
        Assign all galaxies as centrals in the first iteration.
        """
        logging.info("Assigning initial central and satellite galaxies...")
        self.gal_is_central = np.ones(
            len(self.gal_ids), dtype=bool
        )  # All galaxies are considered centrals initially
        self.gal_is_satellite = np.zeros(
            len(self.gal_ids), dtype=bool
        )  # No satellites initially
        self.group_ids = np.arange(
            len(self.gal_ids), dtype="int32"
        )  # Each galaxy is its own group initially
        self.iteration_group_membership[str(self.iteration_counter)] = (
            self.group_ids.copy()
        )
        self.iteration_central_assignment[str(self.iteration_counter)] = (
            self.gal_is_central.copy()
        )
        self.iteration_satellite_assignment[str(self.iteration_counter)] = (
            self.gal_is_satellite.copy()
        )
        logging.info("Initial central and satellite assignment complete.")

    def update_group_luminosity_and_centres(self):
        logging.info(
            "Updating group properites and centres..."
        )

        stellar_mass = self.stellar_mass if self.mass_assignment_mode == "shmr" else np.ones_like(self.gal_luminosities)
        (
            self.unique_groups,
            self.group_centres_ra,
            self.group_centres_dec,
            self.group_centres_z,
            self.group_luminosities,
            self.group_stellar_masses,
            self.group_stellar_mass_3_biggest,
            self.group_bcg_abs_mag,
            self.group_sizes,
            self.group_bcg_is_red,
        ) = brightest_galaxy_centers_fast(
            self.gal_luminosities,
            stellar_mass,
            self.abs_mag,
            self.is_red,
            self.ra,
            self.dec,
            self.zobs,
            self.group_ids,
            self.lf_phi_star,
            self.lf_M_star,
            self.lf_alpha,
            self.mag_limit,
            self.omega_matter,
            self.h,
        )
        logging.info("Group properties and centres updated.")

    def update_group_halo_masses(self):
        logging.info(f"Updating halo masses using {self.mass_assignment_mode} relation...")
        self.group_bcg_k_corrs = k_corr(self.group_centres_z)

        self.group_magnitudes = get_all_luminosity_to_magnitude(
            self.group_luminosities, self.abs_mag_sun
        )

        if self.make_plots:
            plt.hist(np.log10(self.group_luminosities * 1e14), log=True, bins=25)
            plt.title("Group Luminosity Distribution Before Mass Assignment")
            plt.xlabel(r"log10(Group Luminosity [$h^{-2} L_{\odot}$])")
            plt.ylabel("Frequency")
            plt.savefig(
                f"{self.plot_save_dir}/group_luminosities_iter_{self.iteration_counter}_pre_mass_assignment.png"
            )
            plt.clf()
            plt.hist(self.group_bcg_k_corrs)
            plt.title("Group BCG K-Correction Distribution")
            plt.xlabel("K-correction [mag]")
            plt.ylabel("Frequency")
            plt.savefig(
                f"{self.plot_save_dir}/group_bcg_k_corrections_iter_{self.iteration_counter}.png"
            )
            plt.clf()

            plt.hist(self.group_magnitudes, log=True, bins=25)
            plt.title("Group Absolute Magnitude Distribution Before Mass Assignment")
            plt.xlabel("Absolute Magnitude [mag]")
            plt.ylabel("Frequency")
            plt.savefig(
                f"{self.plot_save_dir}/group_magnitudes_iter_{self.iteration_counter}_pre_mass_assignment.png"
            )
            plt.clf()

            plt.hist(self.group_centres_z)
            plt.title("Group Redshift Distribution")
            plt.xlabel("Redshift")
            plt.ylabel("Frequency")
            plt.savefig(
                f"{self.plot_save_dir}/group_redshifts_iter_{self.iteration_counter}.png"
            )
            plt.clf()

        if self.mass_assignment_mode == "shmr":
            if self.shmr_method == "van_Kampen":
                self.group_halo_masses = stellar2halo_mass_van_kampen(self.group_stellar_mass_3_biggest)
            else:
                self.group_halo_masses = linear_stellar_mass2halo_mass(
                    self.group_stellar_masses,
                    self.shmr_intercept,
                    self.shmr_slope,
                )


        elif self.mass_assignment_mode == 'lhmr':
            self.group_halo_masses = linear_luminosity2halo_mass(self.group_luminosities, self.lhmr_intercept, self.lhmr_slope)

        elif self.mass_assignment_mode == 'lhmr_dynamical_calibrated':
            if self.iteration_counter == 0:
                self.group_halo_masses = find_all_initial_mass_to_light(self.group_luminosities, 100.0)
            else:
                self.group_halo_masses = linear_luminosity2halo_mass(
                    self.group_luminosities,
                    self.lhmr_dyn_current_intercept,
                    self.lhmr_dyn_current_slope,
                )

        elif self.mass_assignment_mode == 'red_blue_lhmr':
            self.group_halo_masses = red_blue_linear_luminosity2halo_mass(
                self.group_luminosities,
                self.group_bcg_is_red,
                self.lhmr_intercept_red,
                self.lhmr_slope_red,
                self.lhmr_intercept_blue,
                self.lhmr_slope_blue,
            )
        elif self.mass_assignment_mode == "abundance_match":
            self.group_halo_masses = abundance_match_halo_masses(
                self.group_luminosities,
                self.group_centres_z,
                self.group_bcg_abs_mag,
                self.group_bcg_k_corrs,
                self.mag_limit,
                self.survey_fractional_area,
                self.hmf_masses,
                self.hmf_dlog10m,
                self.omega_matter,
                self.h,
            )

        if self.make_plots:
            plt.hist(self.group_halo_masses, log=True, bins=25)
            plt.title("Group Halo Mass Distribution After Mass Assignment")
            plt.xlabel(r"Group Halo Mass [log10($M_{\odot} h^{-1}$)]")
            plt.ylabel("Frequency")
            plt.savefig(
                f"{self.plot_save_dir}/halo_masses_iter_{self.iteration_counter}_post_mass_assignment.png"
            )
            plt.clf()

        logging.info("Halo masses updated.")

    def apply_halo_finder(self):
        logging.info("Performing iteration of group finder")
        if self.mass_assignment_mode in ("shmr", "lhmr", "red_blue_lhmr"):
            use_active_groups = self.active_group_ids is not None
            if use_active_groups:
                logging.info(
                    "Restricting group updates to %d groups with prior membership changes",
                    len(self.active_group_ids),
                )
            active_group_ids = (
                self.active_group_ids
                if use_active_groups
                else np.empty(0, dtype=np.int64)
            )
        else:
            active_group_ids = np.empty(0, dtype=np.int64)
            use_active_groups = False

        self.new_members, self.gal_is_central, self.gal_is_satellite = (
            update_group_membership_halofinder(
                self.ra,
                self.dec,
                self.zobs,
                self.group_ids,
                self.unique_groups,
                self.group_centres_ra,
                self.group_centres_dec,
                self.group_centres_z,
                self.group_sizes,
                self.group_halo_masses,
                self.gal_kde_tree,
                self.gal_is_central,
                self.gal_is_satellite,
                self.is_red,
                self.red_a_threshold,
                self.red_b_threshold,
                self.blue_a_threshold,
                self.blue_b_threshold,
                self.threshold_b_pivot,
                self.omega_matter,
                self.h,
                active_group_ids,
                use_active_groups,
            )
        )
        logging.info(
            f"Groupfinder iteration complete, number of groups found: {len(np.unique(self.new_members))}"
        )

    def update_lhmr_dynamical_calibration(self):
        logging.info("Updating dynamical LHMR calibration...")
        if self.mass_assignment_mode != "lhmr_dynamical_calibrated":
            return

        group_dynamical_masses = calculate_group_dynamical_masses(
            self.group_ids,
            self.unique_groups,
            self.zobs,
            self.ra,
            self.dec,
            self.group_centres_ra,
            self.group_centres_dec,
            self.group_centres_z,
            self.group_sizes,
            self.lhmr_dyn_A,
            self.omega_matter,
        )

        slope, intercept, n_used = fit_log_luminosity_log_mass_relation(
            self.group_luminosities,
            group_dynamical_masses,
            self.group_sizes,
            self.lhmr_dyn_min_group_members,
        )

        if np.isfinite(slope) and np.isfinite(intercept):
            self.lhmr_dyn_current_slope = slope
            self.lhmr_dyn_current_intercept = intercept
            logging.info(
                "Updated dynamical LHMR calibration using %d groups: slope=%.4f intercept=%.4f",
                n_used,
                slope,
                intercept,
            )
        else:
            logging.warning(
                "Could not update dynamical LHMR calibration (valid groups=%d); retaining previous slope/intercept",
                n_used,
            )

        if self.make_plots:
            # Plot dynamical mass vs group luminosity with the fitted relation
            plt.scatter(
                np.log10(self.group_luminosities * 1e14),
                group_dynamical_masses,
                s=0.1,
            )
            x = np.linspace(
                np.min(self.group_luminosities * 1e14),
                np.max(self.group_luminosities * 1e14),
                100,
            )
            y = 10 ** (self.lhmr_dyn_current_intercept + self.lhmr_dyn_current_slope * np.log10(x))
            plt.plot(np.log10(x), np.log10(y), color="red", label="Fitted LHMR")
            plt.title("Dynamical Mass vs Group Luminosity with Fitted LHMR")
            plt.xlabel(r"log10(Group Luminosity [$h^{-2} L_{\odot}$])")
            plt.ylabel(r"log10(Dynamical Mass [$M_{\odot} h^{-1}$])")
            plt.legend()
            plt.savefig(
                f"{self.plot_save_dir}/dynamical_mass_vs_luminosity_iter_{self.iteration_counter}.png"
            )
            plt.clf()
            


    def save_galaxy_groups(self):
        logging.info(f"Saving galaxy groups at the location: {self.save_path}")
        header = "id_galaxy_sky\tid_finder_group"
        data = np.column_stack((self.gal_ids, self.group_ids))
        np.savetxt(self.save_path, data, header=header)

    def iterate_halo_finder(self):
        self.active_group_ids = None

        while self.iteration_counter < self.max_iterations:
            self.iteration_counter += 1
            logging.info(f"Starting iteration number : {self.iteration_counter}...")
            self.apply_halo_finder()
            changed_group_values = np.sum(self.new_members != self.group_ids)
            logging.info(
                f"{changed_group_values} galaxies have changed group membership last iteration"
            )
            if changed_group_values == 0:
                logging.info(
                    f"Ending finder iteration, no changes in membership detected"
                )
                break
            else:
                changed_mask = self.new_members != self.group_ids
                groups_with_removed_members = self.group_ids[changed_mask]
                groups_with_added_members = self.new_members[changed_mask]
                self.active_group_ids = np.unique(
                    np.concatenate(
                        (groups_with_removed_members, groups_with_added_members)
                    )
                )

                self.iteration_group_membership[str(self.iteration_counter)] = (
                    self.new_members.copy()
                )
                self.group_ids = self.new_members.copy()
                self.update_group_luminosity_and_centres()
                if self.mass_assignment_mode == "lhmr_dynamical_calibrated":
                    self.update_lhmr_dynamical_calibration()
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
        score, e_tot, q_tot = s_score(self.group_ids, self.id_group_sky, 5)
        logging.info(
            f"S-score calculated: {score:.6f}, e_tot: {e_tot:.6f}, q_tot: {q_tot:.6f}"
        )

        self.s_tot = score
        # Save the S-score to a file
        np.savetxt(self.s_tot_save_path + "_B:", [score], header="S-score", fmt="%.6f")

    def debugging_plots(self):
        if not self.make_plots:
            return
        logging.info("Creating debugging plots...")
        # Halo masses
        plt.hist(self.group_halo_masses, log=True, bins=25)
        plt.title("Group Halo Mass Distribution")
        plt.xlabel(r"Group Halo Mass [log10($M_{\odot} h^{-1}$)]")
        plt.ylabel("Frequency")
        plt.savefig(
            f"{self.plot_save_dir}/halo_masses_iter_{self.iteration_counter}.png"
        )
        plt.clf()

        # Halo luminosities
        plt.hist(np.log10(self.group_luminosities * 1e14), log=True, bins=25)
        plt.title("Group Luminosity Distribution")
        plt.xlabel(r"log10(Group Luminosity [$h^{-2} L_{\odot}$])")
        plt.ylabel("Frequency")
        plt.savefig(
            f"{self.plot_save_dir}/halo_luminosities_iter_{self.iteration_counter}.png"
        )
        plt.clf()

        # Halo N
        plt.hist(self.group_sizes, log=True, bins=50)
        plt.title("Group Membership Count Distribution")
        plt.ylabel("Frequency")
        plt.xlabel("Galaxy Count per Group")
        plt.savefig(
            f"{self.plot_save_dir}/halo_population_counts_iter_{self.iteration_counter}.png"
        )
        plt.clf()

        # Halo locations
        plt.scatter(self.group_centres_ra, self.group_centres_dec, s=0.1)
        plt.title("Group Sky Positions")
        plt.xlabel("Right Ascension [deg]")
        plt.ylabel("Declination [deg]")

        plt.savefig(
            f"{self.plot_save_dir}/halo_locations_iter_{self.iteration_counter}.png"
        )
        plt.clf()

        # Halo L vs M

        plt.scatter(
            np.log10(self.group_stellar_masses),
            self.group_halo_masses,
            s=0.1,
        )
        plt.title("Group Stellar Mass vs Halo Mass")
        plt.xlabel(r"log10(Group Stellar Mass [$M_{\odot}$])")
        plt.ylabel(r"Group Halo Mass [log10($M_{\odot} h^{-1}$)]")
        plt.savefig(
            f"{self.plot_save_dir}/halo_m_vs_stellar_m_iter_{self.iteration_counter}.png"
        )
        plt.clf()

        plt.scatter(
            self.group_halo_masses,
            np.log10(self.group_luminosities * 1e14),
            s=0.1,
        )
        plt.title("Group Halo Mass vs Luminosity")
        plt.xlabel(r"Group Halo Mass [log10($M_{\odot} h^{-1}$)]")
        plt.ylabel(r"log10(Group Luminosity [$h^{-2} L_{\odot}$])")

        plt.savefig(
            f"{self.plot_save_dir}/halo_m_vs_l_iter_{self.iteration_counter}.png"
        )
        plt.clf()
        logging.info("Debugging plots created.")
        # Going to plot basically everything here to see if theres a hickup


# ------------------------
# Example Usage
# ------------------------


class RunHaloFinder(HaloFinder):
    def __init__(self, config_reader):
        """
        Initialize HaloFinder with configuration from ConfigReader.

        Args:
            config_reader (ConfigReader): An instance of ConfigReader with loaded configuration
        """
        super().__init__(config_reader)

    def run(self):
        logging.info("Running the Halo Finder...")
        self.load_catalogue_data()
        if self.mass_assignment_mode == "abundance_match":
            self.generate_hmf()
        self.get_all_comoving_distances()
        self.create_KDE_tree()
        self.initial_group_central_satellite_assignment()
        self.initial_luminosities()
        self.update_group_luminosity_and_centres()
        if self.mass_assignment_mode in ("lhmr_dynamical_calibrated"):
            self.lhmr_dyn_current_slope = self.lhmr_slope
            self.lhmr_dyn_current_intercept = self.lhmr_intercept
        self.update_group_halo_masses()
        logging.info(f"Tunable parameters for this run:")
        logging.info(f"Red a threshold: {self.red_a_threshold}")
        logging.info(f"Red b threshold: {self.red_b_threshold}")
        logging.info(f"Blue a threshold: {self.blue_a_threshold}")
        logging.info(f"Blue b threshold: {self.blue_b_threshold}")
        self.iterate_halo_finder()
        if self.run_mock_comparion == True:
            self.s_score()
        elif self.optimse_on_mock == True:
            self.s_score()
        elif self.optimse_parameter_space == True:
            self.s_score()
        logging.info("Halo Finder run complete.")

    def run_cached_cat_hmf_comoving_KDE(self):
        self.initial_group_central_satellite_assignment()
        self.initial_luminosities()
        self.update_group_luminosity_and_centres()
        if self.mass_assignment_mode in ("lhmr_dynamical_calibrated"):
            self.lhmr_dyn_current_slope = self.lhmr_slope
            self.lhmr_dyn_current_intercept = self.lhmr_intercept
        self.update_group_halo_masses()
        logging.info(f"Tunable parameters for this run:")
        logging.info(f"Red a threshold: {self.red_a_threshold}")
        logging.info(f"Red b threshold: {self.red_b_threshold}")
        logging.info(f"Blue a threshold: {self.blue_a_threshold}")
        logging.info(f"Blue b threshold: {self.blue_b_threshold}")
        self.iterate_halo_finder()
        if self.run_mock_comparion == True:
            self.s_score()
        elif self.optimse_on_mock == True:
            self.s_score()
        elif self.optimse_parameter_space == True:
            self.s_score()
        logging.info("Halo Finder run complete.")
