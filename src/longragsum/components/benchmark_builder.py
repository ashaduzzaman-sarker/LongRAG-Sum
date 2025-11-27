# src/longragsum/components/benchmark_builder.py
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets
from pathlib import Path
from longragsum.logging.logger import logger
from tqdm import tqdm
import torch

class LongSum2025Builder:
    def __init__(self, config):
        self.config = config.longsum_2025
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_sample(self, ds_config):
        logger.info(f"Loading {ds_config.name.upper()}...")
        ds = load_dataset(ds_config.hf_name, ds_config.config, split="train")

        def preprocess(example):
            text = example[ds_config.text_key]
            if isinstance(text, list):
                text = " ".join(text)
            summary = example[ds_config.summary_key]
            if isinstance(summary, list):
                summary = summary[0]["text"] if ds_config.name == "booksum" else summary[0]
            return {
                "text": str(text).strip(),
                "summary": str(summary).strip(),
                "source": ds_config.name,
                "domain": ds_config.domain,
                "id": f"{ds_config.name}_{example.get('id', len(ds))}"
            }

        ds = ds.map(preprocess, remove_columns=ds.column_names)

        total = ds_config.train_samples + ds_config.val_samples + ds_config.test_samples
        ds = ds.select(range(min(total, len(ds))))  # safety

        train = ds.select(range(ds_config.train_samples))
        val_start = ds_config.train_samples
        val_end = val_start + ds_config.val_samples
        test_start = val_end
        test_end = test_start + ds_config.test_samples

        return {
            "train": train,
            "validation": ds.select(range(val_start, val_end)),
            "test": ds.select(range(test_start, test_end))
        }

    def build_raw(self):
        logger.info("Building LongSum-2025 raw benchmark...")
        all_splits = {"train": [], "validation": [], "test": []}

        for ds_config in self.config.datasets:
            try:
                split_data = self._load_and_sample(ds_config)
                for split in ["train", "validation", "test"]:
                    all_splits[split].append(split_data[split])
            except Exception as e:
                logger.error(f"Failed to load {ds_config.name}: {e}")

        merged = DatasetDict({
            split: concatenate_datasets(datasets) if datasets else None
            for split, datasets in all_splits.items()
        })

        merged.save_to_disk(self.output_dir / "raw")
        logger.success(f"LongSum-2025 RAW saved: "
                       f"Train={len(merged['train'])}, "
                       f"Val={len(merged['validation'])}, "
                       f"Test={len(merged['test'])}")
        return merged

    def verify_with_bertscore(self, dataset_dict):
        logger.info("Verifying summaries with BERTScore...")
        from bert_score import score

        verified = {"train": [], "validation": [], "test": []}
        threshold = self.config.bertscore_threshold

        for split in ["train", "validation", "test"]:
            ds = dataset_dict[split]
            texts = ds["text"]
            summaries = ds["summary"]

            P, R, F1 = score(summaries, texts, lang="en", verbose=False, batch_size=16)
            mask = (F1 > threshold).tolist()

            verified[split] = ds.select([i for i, keep in enumerate(mask) if keep])
            logger.info(f"{split.upper()}: {sum(mask)}/{len(mask)} passed BERTScore > {threshold}")

        final = DatasetDict(verified)
        final.save_to_disk(self.output_dir / "verified")
        logger.success(f"LongSum-2025 VERIFIED saved to {self.output_dir / 'verified'}")
        return final

    def push_to_hub(self, dataset_dict, repo_id="ashaduzzaman/LongSum-2025"):
        logger.info(f"Pushing to Hugging Face Hub: {repo_id}")
        dataset_dict.push_to_hub(repo_id, private=False)
        logger.success(f"LongSum-2025 is LIVE: https://huggingface.co/datasets/{repo_id}")