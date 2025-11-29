# src/longragsum/components/reader.py
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, LoraConfig
from typing import List, Dict, Optional, Tuple
import logging
from longragsum.logging.logger import logger

class QwenReader:
    def __init__(self, model_name: str, use_lora: bool = True, load_4bit: bool = True):
        self.model_name = model_name
        self.use_lora = use_lora
        self.load_4bit = load_4bit
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._setup_model()
    
    def _setup_model(self):
        """Initialize Qwen2.5 model with optimal quantization"""
        logger.info(f"Loading Qwen2.5 model: {self.model_name}")
        
        # 4-bit quantization for T4
        if self.load_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup pipeline for easy inference
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        logger.success(f"Qwen2.5 reader loaded: {self.model_name}")
    
    def apply_lora(self, lora_config: dict, lora_path: Optional[str] = None):
        """Apply LoRA adapters"""
        if not self.use_lora:
            return
        
        if lora_path and lora_path.exists():
            # Load existing LoRA
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            logger.success(f"Loaded LoRA from {lora_path}")
        else:
            # Create new LoRA
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, lora_config)
            logger.success("Created new LoRA configuration")
    
    def generate_summary(
        self, 
        context_chunks: List[str], 
        query: str = "Summarize the following document:",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate summary using RAG context"""
        
        # Format RAG prompt for Qwen2.5
        context = "\n\n".join([f"Passage {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""<|im_start|>system
You are an expert summarizer. Create a comprehensive, coherent summary that captures all key points from the retrieved passages. Focus on factual accuracy and complete coverage.

<|im_end|>
<|im_start|>user
{query}

Context:
{context}

<|im_end|>
<|im_start|>assistant
"""
        
        # Generate with optimized parameters
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        summary = outputs[0]['generated_text'].strip()
        
        # Clean Qwen output
        if summary.startswith("```"):
            summary = summary.split("```")[1].strip()
        
        return summary
    
    def batch_generate(self, contexts: List[List[str]], queries: List[str], **kwargs) -> List[str]:
        """Batch generation for evaluation"""
        results = []
        for context_chunks, query in zip(contexts, queries):
            summary = self.generate_summary(context_chunks, query, **kwargs)
            results.append(summary)
        return results