# src/longragsum/data_processor.py
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import torch
from typing import Dict, List
from longragsum.logging.logger import logger

class LongSumDataProcessor:
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_rag_training_example(self, example: Dict) -> Dict:
        """Format single example for RAG training"""
        document = example['text']
        summary = example['summary']
        domain = example['domain']
        
        # Qwen2.5 RAG prompt template
        prompt = f"""<|im_start|>system
You are an expert summarizer specializing in {domain} documents. Create comprehensive, accurate summaries that capture all key information from the provided context.

<|im_end|>
<|im_start|>user
Document: {document}

Please provide a comprehensive summary:
<|im_end|>
<|im_start|>assistant
{summary}<|im_end|>"""
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=16384,  # Qwen2.5 safe limit
            padding=False,
            return_tensors="pt"
        )
        
        # Create labels (shift input_ids)
        labels = tokenized['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': labels
        }
    
    def process_dataset(self, dataset: Dataset, max_samples: int = None) -> Dataset:
        """Process full dataset for training"""
        logger.info(f"Processing {len(dataset)} examples...")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Apply formatting
        processed = dataset.map(
            self.format_rag_training_example,
            remove_columns=dataset.column_names,
            desc="Formatting RAG examples"
        )
        
        logger.success(f"Processed {len(processed)} training examples")
        return processed