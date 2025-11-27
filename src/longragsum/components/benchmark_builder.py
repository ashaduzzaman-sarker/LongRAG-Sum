# src/longragsum/components/benchmark_builder.py
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from pathlib import Path
from longragsum.logging.logger import logger
from tqdm import tqdm
import re

class LongSum2025Builder:
    def __init__(self, config):
        self.config = config.longsum_2025
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_empty_dataset(self):
        """Create empty dataset compatible with all HF versions"""
        return Dataset.from_dict({
            "text": [], 
            "summary": [], 
            "source": [], 
            "domain": [], 
            "id": []
        })

    def _load_and_sample(self, ds_config):
        """Load dataset and safely sample with error handling"""
        logger.info(f"Loading {ds_config['name'].upper()}...")
        
        try:
            # Handle datasets with/without config
            if ds_config['config'] is None or ds_config['config'] == 'null':
                ds = load_dataset(ds_config['hf_name'], split="train")
            else:
                ds = load_dataset(ds_config['hf_name'], ds_config['config'], split="train")
            
            def preprocess(example):
                # Handle text field (list or string)
                text = example[ds_config['text_key']]
                if isinstance(text, list):
                    text = " ".join([str(t) for t in text if t])
                text = str(text).strip()
                
                # Clean noise for QMSum
                if ds_config['name'] == 'qmsum':
                    text = re.sub(r'\{[a-zA-Z]+\}', '', text)
                
                # Handle summary field (list, dict, or string)
                summary = example[ds_config['summary_key']]
                if isinstance(summary, list):
                    if len(summary) > 0:
                        if isinstance(summary[0], dict) and 'text' in summary[0]:
                            summary = summary[0]['text']
                        else:
                            summary = str(summary[0])
                    else:
                        summary = ""
                elif isinstance(summary, dict) and 'text' in summary:
                    summary = summary['text']
                summary = str(summary).strip()
                
                return {
                    "text": text,
                    "summary": summary,
                    "source": ds_config['name'],
                    "domain": ds_config['domain'],
                    "id": f"{ds_config['name']}_{str(example.get('id', example.get('idx', '')))}"
                }

            ds = ds.map(preprocess, remove_columns=ds.column_names)
            
            # Safe sampling with bounds checking
            total_needed = (ds_config['train_samples'] + 
                          ds_config['val_samples'] + 
                          ds_config['test_samples'])
            available = len(ds)
            
            if total_needed > available:
                logger.warning(f"{ds_config['name']}: Need {total_needed}, only {available} available")
                total_needed = available
            
            ds_sampled = ds.select(range(total_needed))
            
            # Split safely
            train_end = min(ds_config['train_samples'], len(ds_sampled))
            val_end = min(train_end + ds_config['val_samples'], len(ds_sampled))
            test_end = min(val_end + ds_config['test_samples'], len(ds_sampled))
            
            splits = {
                "train": ds_sampled.select(range(0, train_end)) if train_end > 0 else self._create_empty_dataset(),
                "validation": ds_sampled.select(range(train_end, val_end)) if val_end > train_end else self._create_empty_dataset(),
                "test": ds_sampled.select(range(val_end, test_end)) if test_end > val_end else self._create_empty_dataset()
            }
            
            logger.info(f"{ds_config['name']}: Train={len(splits['train'])}, Val={len(splits['validation'])}, Test={len(splits['test'])}")
            return splits
            
        except Exception as e:
            logger.error(f"Failed to load {ds_config['name']}: {str(e)}")
            return {
                "train": self._create_empty_dataset(),
                "validation": self._create_empty_dataset(),
                "test": self._create_empty_dataset()
            }

    def build_raw(self):
        """Build raw LongSum-2025 dataset with all domains"""
        logger.info("🚀 Building LongSum-2025 raw benchmark...")
        all_splits = {"train": [], "validation": [], "test": []}

        successful_datasets = 0
        for ds_config in self.config['datasets']:
            try:
                split_data = self._load_and_sample(ds_config)
                for split in ["train", "validation", "test"]:
                    if len(split_data[split]) > 0:
                        all_splits[split].append(split_data[split])
                successful_datasets += 1
            except Exception as e:
                logger.error(f"Error processing {ds_config['name']}: {e}")

        # Merge successful splits
        merged = DatasetDict({
            split: concatenate_datasets(datasets) if len(datasets) > 0 else self._create_empty_dataset()
            for split, datasets in all_splits.items()
        })

        # Save raw dataset
        raw_path = self.output_dir / "raw"
        raw_path.mkdir(exist_ok=True)
        merged.save_to_disk(raw_path)
        
        stats = {
            "train": len(merged["train"]),
            "validation": len(merged["validation"]), 
            "test": len(merged["test"]),
            "successful_domains": successful_datasets
        }
        
        logger.success(f"✅ LongSum-2025 RAW saved to {raw_path}")
        logger.success(f"📊 Stats: Train={stats['train']}, Val={stats['validation']}, Test={stats['test']}, Domains={stats['successful_domains']}")
        
        return merged, stats

    def verify_with_bertscore(self, dataset_dict):
        """Verify summary quality using BERTScore"""
        logger.info("🔍 Verifying summaries with BERTScore (threshold=0.72)...")
        
        try:
            from bert_score import score
        except ImportError:
            logger.warning("bert-score not installed. Skipping verification...")
            return dataset_dict

        verified = {"train": [], "validation": [], "test": []}
        threshold = self.config['bertscore_threshold']

        for split_name in ["train", "validation", "test"]:
            ds = dataset_dict[split_name]
            if len(ds) == 0:
                logger.warning(f"{split_name} split is empty, skipping verification")
                verified[split_name] = ds
                continue

            # Extract Python lists
            summaries_list = ds["summary"]
            texts_list = ds["text"]
            
            # Filter out empty summaries
            valid_mask = []
            valid_summaries = []
            valid_texts = []
            valid_indices = []
            
            for i, (s, t) in enumerate(zip(summaries_list, texts_list)):
                s_clean = str(s).strip()
                t_clean = str(t).strip()
                if s_clean != "" and t_clean != "" and len(s_clean) > 10:
                    valid_mask.append(True)
                    valid_summaries.append(s_clean)
                    valid_texts.append(t_clean)
                    valid_indices.append(i)
                else:
                    valid_mask.append(False)
            
            if len(valid_summaries) == 0:
                logger.warning(f"No valid summaries in {split_name}")
                verified[split_name] = ds
                continue

            logger.info(f"Verifying {split_name} ({len(valid_summaries)} valid examples)...")
            
            # Compute BERTScore
            try:
                P, R, F1 = score(
                    valid_summaries, 
                    valid_texts, 
                    lang="en", 
                    verbose=False, 
                    batch_size=8
                )
                scores = F1.tolist()
            except Exception as e:
                logger.warning(f"BERTScore failed for {split_name}: {e}. Keeping all examples.")
                verified[split_name] = ds
                continue
            
            # Select passing examples
            passing_indices = [valid_indices[i] for i, score in enumerate(scores) if score >= threshold]
            verified[split_name] = ds.select(passing_indices) if passing_indices else self._create_empty_dataset()
            
            passed = len(passing_indices)
            total_valid = len(valid_indices)
            logger.info(f"{split_name.upper()}: {passed}/{total_valid} passed (threshold={threshold:.3f})")

        final_verified = DatasetDict(verified)
        
        # Save verified dataset
        verified_path = self.output_dir / "verified"
        verified_path.mkdir(exist_ok=True)
        final_verified.save_to_disk(verified_path)
        
        logger.success(f"✅ LongSum-2025 VERIFIED saved to {verified_path}")
        return final_verified

    def push_to_hub(self, dataset_dict, repo_id="ashaduzzaman/LongSum-2025"):
        """Push verified dataset to Hugging Face Hub"""
        logger.info(f"📤 Pushing LongSum-2025 to Hugging Face: {repo_id}")
        
        try:
            dataset_dict.push_to_hub(repo_id, private=False)
            logger.success(f"🎉 LongSum-2025 is LIVE: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            logger.error(f"Failed to push to HF: {e}")
            logger.info("Dataset saved locally")