# src/longragsum/utils/common.py
import os
import yaml
from pathlib import Path
from box import ConfigBox
from ensure import ensure_annotations

@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    """Read yaml file and return ConfigBox (allows dot access)"""
    with open(path, "r") as f:
        content = yaml.safe_load(f)
        return ConfigBox(content)

@ensure_annotations
def create_directories(path_list: list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

@ensure_annotations
def save_json(data: dict, path: Path):
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=4)