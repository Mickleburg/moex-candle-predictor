"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Dictionary with configuration parameters.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config file is invalid.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    return config


def load_all_configs(config_dir: str | Path = "configs") -> dict[str, dict]:
    """Load all YAML configuration files from directory.
    
    Args:
        config_dir: Path to configs directory (relative to ml/).
        
    Returns:
        Dictionary mapping config names to config dicts.
    """
    config_dir = Path(config_dir)
    
    configs = {}
    
    # Expected config files
    config_files = ["data.yaml", "features.yaml", "train.yaml", "eval.yaml"]
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            config_name = config_file.replace(".yaml", "")
            configs[config_name] = load_config(config_path)
        else:
            print(f"Warning: Config file not found: {config_path}")
    
    return configs


if __name__ == "__main__":
    # Test config loading
    configs = load_all_configs("../../configs")
    for name, config in configs.items():
        print(f"\n{name}:")
        print(config)
