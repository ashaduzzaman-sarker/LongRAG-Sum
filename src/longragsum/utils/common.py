import os
import yaml
from pathlib import Path

def read_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_directories(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)

def save_json(data, path):
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=4)