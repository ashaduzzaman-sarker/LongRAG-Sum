# src/longragsum/data_processor.py - FIXED VERSION
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Dict, List
from longragsum.logging.logger import logger
import torch

class LongSumDataProcessor:
    def __init__(self, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = 8192  # Conservative for T4
    
    def format_rag_training_example(self, example: Dict) -> Dict:
        """Format single example for RAG training - FIXED FOR TRAINER"""
        document = str(example['text'])[:8000]  # Truncate long docs
        summary = str(example['summary'])
        domain = str(example['domain'])
        
        # Qwen2.5 RAG prompt template (instruction format)
        prompt = f"""<|im_start|>system
You are an expert summarizer specializing in {domain} documents. Create comprehensive, accurate summaries.
<|im_end|>
<|im_start|>user
Document: {document}
Please provide a comprehensive summary:
<|im_end|>
<|im_start|>assistant
{summary}<|im_end|>"""
        
        # Tokenize WITHOUT padding/truncation - let Trainer handle it
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  
            add_special_tokens=True
        )
        
        # Create labels by copying input_ids
        input_ids = tokenized['input_ids']
        labels = [-100] * len(input_ids)  # Default: ignore all tokens
        
        # Set labels for summary part only (after "assistant")
        assistant_start = prompt.find("<|im_start|>assistant")
        if assistant_start != -1:
            # Find tokens corresponding to assistant response
            assistant_tokens_start = len(self.tokenizer.encode(prompt[:assistant_start]))
            for i in range(assistant_tokens_start, len(input_ids)):
                labels[i] = input_ids[i]  # Keep original tokens as labels
        
        return {
            'input_ids': input_ids,           
            'attention_mask': tokenized['attention_mask'],  
            'labels': labels                  
        }
    
    def process_dataset(self, dataset: Dataset, max_samples: int = None) -> Dataset:
        """Process full dataset for training - FIXED"""
        logger.info(f"Processing {len(dataset)} examples...")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        def batch_format(examples):
            """Batch processing for efficiency"""
            texts = []
            summaries = []
            domains = []
            
            for text, summary, domain in zip(examples['text'], examples['summary'], examples['domain']):
                texts.append(str(text)[:8000])
                summaries.append(str(summary))
                domains.append(str(domain))
            
            # Tokenize entire batch
            prompts = []
            for text, summary, domain in zip(texts, summaries, domains):
                prompt = f"""<|im_start|>system
You are an expert summarizer specializing in {domain} documents. Create comprehensive, accurate summaries.
<|im_end|>
<|im_start|>user
Document: {text}
Please provide a comprehensive summary:
<|im_end|>
<|im_start|>assistant
{summary}<|im_end|>"""
                prompts.append(prompt)
            
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_length,
                padding=False,  
                return_tensors=None
            )
            
            # Create labels
            input_ids = tokenized['input_ids']
            labels_list = []
            
            for i, prompt in enumerate(prompts):
                labels = [-100] * len(input_ids[i])
                # Find assistant response start
                assistant_start = prompt.find("<|im_start|>assistant")
                if assistant_start != -1:
                    assistant_tokens_start = len(self.tokenizer.encode(prompt[:assistant_start]))
                    for j in range(assistant_tokens_start, len(input_ids[i])):
                        labels[j] = input_ids[i][j]
                labels_list.append(labels)
            
            return {
                'input_ids': input_ids,
                'attention_mask': tokenized['attention_mask'],
                'labels': labels_list
            }
        
        # Process in batches for memory efficiency
        processed = dataset.map(
            batch_format,
            batched=True,
            batch_size=8,
            remove_columns=dataset.column_names,
            desc="Formatting RAG examples"
        )
        
        logger.success(f"Processed {len(processed)} training examples")
        return processed