import os
from datasets import load_dataset
from longragsum.logging.logger import logger
from pathlib import Path
import json

class LongFormDataIngestion:
    def __init__(self, config):
        self.config = config

    def download_and_save(self):
        os.makedirs(self.config.raw_data_dir, exist_ok=True)
        save_path = self.config.raw_data_dir / "booksum.parquet"

        if save_path.exists():
            logger.info(f"Data already exists at {save_path}")
            return save_path

        logger.info(f"Downloading {self.config.dataset_name} – {self.config.split}")
        ds = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.split,
            trust_remote_code=True
        )

        # Keep only very long examples for true long-form research
        def is_long(example):
            return len(example["chapter"].split()) > 3000  # >~10k tokens

        ds_long = ds.filter(is_long, num_proc=4)
        logger.info(f"Filtered to {len(ds_long)} long documents")

        ds_long.to_parquet(save_path)
        logger.success(f"Saved raw long-form data → {save_path}")
        return save_path