import yaml
import pytest

from utils import ConfigReader


def test_config_reader_loads_sections(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_options": {"run_hmf": True, "run_lf": False},
                "cosmology": {"omega_matter": 0.3},
                "survey_fractional_sky_area": 0.25,
                "column_names": {"ra": "RA"},
                "file_locations": {"output": "out.csv"},
                "finder_options": {"threads": 2},
                "threshold_model_params": {
                    "red_a_threshold": 1.0,
                    "red_b_threshold": -0.5,
                    "blue_a_threshold": 1.2,
                    "blue_b_threshold": -0.3,
                    "threshold_b_pivot": 13.0,
                    "completeness_coefficient": 1.5,
                },
                "abundance_match_params": {"m_min": 10.0},
                "shmr_params": {"shmr_slope": 1.1, "shmr_intercept": -0.2},
                "lhmr_params": {"lhmr_slope": 1.3, "lhmr_intercept": -3.0},
                "red_blue_lhmr_params": {
                    "lhmr_slope_red": 1.4,
                    "lhmr_intercept_red": -3.1,
                    "lhmr_slope_blue": 1.2,
                    "lhmr_intercept_blue": -2.9,
                },
                "lhmr_dynamical_calibrated_params": {"A": 1.5, "min_group_members": 6},
                "mock_comparison_options": {"min_group_size": 2},
            }
        )
    )

    reader = ConfigReader(str(config_path))
    config = reader.load_config()

    assert config["cosmology"]["omega_matter"] == 0.3
    assert reader.get_run_options() == {"run_hmf": True, "run_lf": False}
    assert reader.get_cosmology() == {"omega_matter": 0.3}
    assert reader.get_survey_fractional_area() == 0.25
    assert reader.get_column_names() == {"ra": "RA"}
    assert reader.get_file_locations() == {"output": "out.csv"}
    assert reader.get_setup_options() == {"threads": 2}
    assert reader.get_finder_options() == {"threads": 2}
    assert reader.get_hmf_options() == {"m_min": 10.0}
    assert reader.get_threshold_model_params()["red_a_threshold"] == 1.0
    assert reader.get_threshold_model_params()["completeness_coefficient"] == 1.5
    assert reader.get_shmr_params()["shmr_slope"] == 1.1
    assert reader.get_lhmr_params()["lhmr_intercept"] == -3.0
    assert reader.get_red_blue_lhmr_params()["lhmr_slope_red"] == 1.4
    assert reader.get_lhmr_dynamical_calibrated_params()["A"] == 1.5
    assert reader.get_mock_comparison_options() == {"min_group_size": 2}
    assert reader.should_run_module("hmf") is True
    assert reader.should_run_module("lf") is False
    assert reader.should_run_module("missing") is False


def test_config_reader_missing_file(tmp_path):
    reader = ConfigReader(str(tmp_path / "missing.yaml"))

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        reader.load_config()


def test_config_reader_invalid_yaml(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("run_options: [")

    reader = ConfigReader(str(config_path))

    with pytest.raises(yaml.YAMLError, match="Error parsing YAML file"):
        reader.load_config()


def test_config_reader_legacy_aliases_and_defaults(tmp_path):
    config_path = tmp_path / "legacy_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_options": {"run_group_finder": True},
                "cosmology": {"omega_matter": 0.27},
                "survey_fractional_sky_area": 0.1,
                "column_names": {"z": "Z"},
                "file_locations": {"catalogue": "catalog.csv"},
                "setup_options": {
                    "red_a_threshold": 0.1,
                    "red_b_threshold": 0.2,
                    "blue_a_threshold": 0.3,
                    "blue_b_threshold": 0.4,
                    "threshold_b_pivot": 12.0,
                    "shmr_slope": 1.2,
                    "shmr_intercept": -0.2,
                    "lhmr_slope": 0.8,
                    "lhmr_intercept": 12.5,
                    "lhmr_slope_red": 0.9,
                    "lhmr_intercept_red": 12.7,
                    "lhmr_slope_blue": 0.7,
                    "lhmr_intercept_blue": 12.2,
                },
                "hmf_options": {"m_min": 11.0},
                "mock_comparison_options": {"min_group_size": 3},
                "luminosity_function_options": {"survey_mag_limit": 19.8},
            }
        )
    )

    reader = ConfigReader(str(config_path))
    reader.load_config()

    assert reader.get_setup_options()["red_a_threshold"] == 0.1
    assert reader.get_hmf_options() == {"m_min": 11.0}
    assert reader.get_threshold_model_params()["threshold_b_pivot"] == 12.0
    assert reader.get_shmr_params()["shmr_slope"] == 1.2
    assert reader.get_lhmr_params()["lhmr_intercept"] == 12.5
    assert reader.get_red_blue_lhmr_params()["lhmr_intercept_blue"] == 12.2
    assert reader.get_lhmr_dynamical_calibrated_params() == {}
    assert reader.get_lf_options() == {"survey_mag_limit": 19.8}
    assert reader.should_run_module("group_finder") is True


def test_config_reader_not_implemented_methods(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"run_options": {}}))
    reader = ConfigReader(str(config_path))
    reader.load_config()

    with pytest.raises(NotImplementedError):
        reader.validate_config()
    with pytest.raises(NotImplementedError):
        reader.print_config_summary()
