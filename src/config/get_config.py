import os
import yaml
from config.config_template import ConfigTemplate


def get_config(path: str) -> ConfigTemplate:
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    # Enforce a strict 1:1 match
    expected_keys = set(ConfigTemplate.model_fields.keys())
    actual_keys = set(config_data.keys())
    if expected_keys != actual_keys:
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        error_msg = f"Missing keys: {missing_keys}, Extra keys: {extra_keys}"
        raise Exception(error_msg)
    # Create and validate config
    config = ConfigTemplate(**config_data)
    # Expand environment variables
    config.data_dir = os.path.expandvars(config.data_dir)
    config.project_directory = os.path.expandvars(config.project_directory)
    return config
