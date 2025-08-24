import os
import yaml

current_directory = os.getcwd()
# path for config yaml
config_path = os.path.join(current_directory, "config", "config.yaml")

with open(config_path, "r", encoding="utf-8") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise exc