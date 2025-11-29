# src/longragsum/data/ingestion.py
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict
from longragsum.logging.logger import logger
from pathlib import Path
import re

class DataIngestion:
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = self.config['data']['longsum_2025']['dataset_name']
        self.output_dir = Path(self.config['paths']['output_dir']) / "processed_longsum"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['dev_reader'], trust_remote_code=True)
        self.chunk_size = self.config['data']['longsum_2025']['chunk_size']
        self.chunk_overlap = self.config['data']['longsum_2025']['chunk_overlap']
    
    def load_dataset(self) -> DatasetDict:
        """Load the full dataset from HF"""
        logger.info(f"Loading dataset from HF: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name)
        logger.success(f"Loaded splits: {list(dataset.keys())}")
        return dataset
    
    def clean_text(self, text: str) -> str:
        """Basic cleaning: remove extra spaces, special chars"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s.,?!-]', '', text)
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk long text with overlap"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk))
            if i + self.chunk_size >= len(tokens):
                break
        return chunks
    
    def preprocess_split(self, ds: Dataset, split_name: str) -> Dataset:
        """Preprocess single split"""
        logger.info(f"Preprocessing {split_name} split ({len(ds)} examples)...")
        
        def preprocess_example(example):
            # Clean text and summary
            text = self.clean_text(example['text'])
            summary = self.clean_text(example['summary'])
            
            # Filter empty
            if not text or not summary:
                return None
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Add tokenized length for verification
            token_len = len(self.tokenizer.encode(text))
            
            return {
                'id': example['id'],
                'text': text,
                'summary': summary,
                'domain': example['domain'],
                'source': example['source'],
                'chunks': chunks,
                'token_length': token_len
            }
        
        processed = ds.map(
            preprocess_example,
            remove_columns=ds.column_names,
            desc=f"Preprocessing {split_name}",
            num_proc=4  # Parallel processing
        )
        
        # Remove None entries
        processed = processed.filter(lambda x: x is not None)
        
        logger.success(f"✅ {split_name}: {len(processed)} processed examples")
        return processed
    
    def run_preprocessing(self) -> DatasetDict:
        """Load and preprocess all splits"""
        raw_dataset = self.load_dataset()
        processed = DatasetDict({
            'train': self.preprocess_split(raw_dataset['train'], 'train'),
            'validation': self.preprocess_split(raw_dataset['validation'], 'validation'),
            'test': self.preprocess_split(raw_dataset['test'], 'test')
        })
        
        # Save locally
        processed.save_to_disk(self.output_dir)
        logger.success(f"✅ Preprocessed dataset saved to {self.output_dir}")
        
        return processed