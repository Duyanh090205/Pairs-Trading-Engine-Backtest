"""
Configuration parsing using yaml
"""
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file and return the parsed dict.

    Raises FileNotFoundError if the path does not exist.
    Raises yaml.YAMLError if the file is malformed.
    No key validation is performed here — the caller owns that.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
