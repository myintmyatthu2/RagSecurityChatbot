import yaml
import os

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) #
        return config
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {exc}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading config: {e}")

