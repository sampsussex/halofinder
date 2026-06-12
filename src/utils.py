import yaml
from pathlib import Path
from typing import Dict, List, Any


class ConfigReader:
    """
    A class to read and parse YAML configuration files for astronomical data processing.
    """

    def __init__(self, config_path: str):
        """
        Initialize the config reader with a path to the YAML file.

        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = None

    def load_config(self) -> Dict[str, Any]:
        """
        Load the YAML configuration file.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If there's an error parsing the YAML
        """
        try:
            with open(self.config_path, "r") as file:
                self.config = yaml.safe_load(file)
                return self.config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    def get_run_options(self) -> Dict[str, bool]:
        """Get all run options."""
        return self.config["run_options"]

    def get_cosmology(self) -> Dict[str, float]:
        """Get HMF (Halo Mass Function) options."""
        return self.config["cosmology"]

    def get_survey_fractional_area(self) -> float:
        """Get the region limits (RA/Dec boundaries)."""
        return self.config["survey_fractional_sky_area"]

    def get_column_names(self) -> Dict[str, str]:
        """Get the column name mappings."""
        return self.config["column_names"]

    def get_file_locations(self) -> Dict[str, str]:
        """Get all file output locations."""
        return self.config["file_locations"]

    def get_setup_options(self) -> Dict[str, Any]:
        """Get general finder setup options (legacy alias for finder_options)."""
        if "finder_options" in self.config:
            return self.config["finder_options"]
        return self.config["setup_options"]

    def get_finder_options(self) -> Dict[str, Any]:
        """Get general finder setup options."""
        return self.get_setup_options()

    def get_hmf_options(self) -> Dict[str, float]:
        """Get abundance-matching/HMF options (legacy alias supported)."""
        if "abundance_match_params" in self.config:
            return self.config["abundance_match_params"]
        return self.config["hmf_options"]


    def get_threshold_model_params(self) -> Dict[str, float]:
        """Get group-assignment threshold model parameters."""
        if "threshold_model_params" in self.config:
            return self.config["threshold_model_params"]
        setup = self.get_setup_options()
        return {
            "red_a_threshold": setup["red_a_threshold"],
            "red_b_threshold": setup["red_b_threshold"],
            "blue_a_threshold": setup["blue_a_threshold"],
            "blue_b_threshold": setup["blue_b_threshold"],
            "threshold_b_pivot": setup["threshold_b_pivot"],
        }

    def get_shmr_params(self) -> Dict[str, float]:
        """Get SHMR relation parameters."""
        if "shmr_params" in self.config:
            return self.config["shmr_params"]
        setup = self.get_setup_options()
        return {
            "shmr_slope": setup["shmr_slope"],
            "shmr_intercept": setup["shmr_intercept"],
        }

    def get_lhmr_params(self) -> Dict[str, float]:
        """Get LHMR relation parameters."""
        if "lhmr_params" in self.config:
            return self.config["lhmr_params"]
        setup = self.get_setup_options()
        return {
            "lhmr_slope": setup["lhmr_slope"],
            "lhmr_intercept": setup["lhmr_intercept"],
        }

    def get_red_blue_lhmr_params(self) -> Dict[str, float]:
        """Get red/blue LHMR relation parameters."""
        if "red_blue_lhmr_params" in self.config:
            return self.config["red_blue_lhmr_params"]
        setup = self.get_setup_options()
        return {
            "lhmr_slope_red": setup["lhmr_slope_red"],
            "lhmr_intercept_red": setup["lhmr_intercept_red"],
            "lhmr_slope_blue": setup["lhmr_slope_blue"],
            "lhmr_intercept_blue": setup["lhmr_intercept_blue"],
        }

    def get_lhmr_dynamical_calibrated_params(self) -> Dict[str, Any]:
        """Get parameters for dynamical-mass-calibrated LHMR mode."""
        return self.config.get("lhmr_dynamical_calibrated_params", {})

    def get_mock_comparison_options(self) -> Dict[str, Any]:
        """Get bijective matching options."""
        return self.config["mock_comparison_options"]

    def get_lf_options(self) -> Dict[str, float]:
        """Get luminosity function options."""
        return self.config["luminosity_function_options"]

    def should_run_module(self, module_name: str) -> bool:
        """
        Check if a specific module should be run.

        Args:
            module_name (str): Name of the module (e.g., 'hmf', 'lf', 'group_finder')

        Returns:
            bool: True if the module should be run
        """
        run_key = f"run_{module_name}"
        return self.config["run_options"].get(run_key, False)

    def validate_config(self) -> bool:
        raise NotImplementedError("Configuration validation not implemented yet.")

    def print_config_summary(self):
        raise NotImplementedError("Configuration summary printing not implemented yet.")
