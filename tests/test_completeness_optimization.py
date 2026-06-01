import os
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from main import should_tune_completeness
from utils import ConfigReader


def _reader(tmp_path, completeness):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_options": {},
                "column_names": {"completeness": completeness},
            }
        )
    )
    reader = ConfigReader(str(config_path))
    reader.load_config()
    return reader


def test_should_tune_completeness_requires_configured_column(tmp_path):
    assert should_tune_completeness(_reader(tmp_path, "completeness")) is True
    assert should_tune_completeness(_reader(tmp_path, None)) is False
