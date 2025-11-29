# scripts/train_qwen.py
#!/usr/bin/env python3
"""
Complete Qwen2.5-7B LoRA Training Pipeline for LongSum-2025
Expected: 42-45 ROUGE-L after 2-3 hours on T4
"""

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from longragsum.data_processor import LongSumDataProcessor
from longragsum.logging.logger import logger
import wandb
from datetime import datetime

def main():
    # === CONFIGURATION ===
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    DATASET_NAME = "ashaduzzaman/LongSum-2025"
    OUTPUT_DIR = Path("artifacts/qwen-longsum-lora-v1")
    
    # Training config
    TRAIN_SAMPLES = 4000  # Fast dev training
    EVAL_SAMPLES = 300
    MAX_LENGTH = 8192     # Conservative for T4
    EPOCHS = 3
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    LR = 2e-4
    
    # === STEP 1: LOAD DATASET ===
    logger.info("Loading LongSum-2025 dataset...")
    dataset = load_dataset(DATASET_NAME)
    
    train_ds = dataset["train"].select(range(TRAIN_SAMPLES))
    eval_ds = dataset["validation"].select(range(EVAL_SAMPLES))
    
    logger.success(f"Dataset loaded: {len(train_ds)} train, {len(eval_ds)} eval")
    
    # === STEP 2: PROCESS DATA ===
    logger.info("Processing dataset for RAG training...")
    processor = LongSumDataProcessor(MODEL_NAME)
    
    train_processed = processor.process_dataset(train_ds, TRAIN_SAMPLES)
    eval_processed = processor.process_dataset(eval_ds, EVAL_SAMPLES)
    
    # === STEP 3: LOAD MODEL WITH 4-BIT QUANTIZATION ===
    logger.info("Loading Qwen2.5-7B with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"  # Faster inference
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # === STEP 4: APPLY LoRA ===
    logger.info("Applying LoRA adapters...")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model = prepare_model_for_kbit_training(model)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # === STEP 5: TRAINING SETUP ===
    logger.info("Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=100,
        logging_steps=25,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        learning_rate=LR,
        fp16=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard", "wandb"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )
    
    # === STEP 6: CREATE TRAINER ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=eval_processed,
        tokenizer=tokenizer,
    )
    
    # === STEP 7: START TRAINING ===
    logger.success("🚀 STARTING TRAINING!")
    logger.success(f"📈 Expected completion: ~2-3 hours on T4")
    logger.success(f"🎯 Target: 42-45 ROUGE-L on LongSum-2025 test set")
    
    # Optional: Initialize wandb
    wandb.init(
        project="longrag-sum-qwen25",
        name=f"longsum-v1-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config={
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "train_samples": len(train_processed),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE * GRAD_ACCUM,
            "learning_rate": LR
        }
    )
    
    # Train!
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # === STEP 8: FINAL RESULTS ===
    logger.success("TRAINING COMPLETE!")
    logger.success(f"Model saved to: {OUTPUT_DIR}")
    logger.success("Next steps:")
    logger.success("1. python scripts/evaluate_longsum.py")
    logger.success("2. Compare with baselines (Qwen72B, Llama70B)")
    logger.success("3. Generate paper tables")
    
    # Expected metrics
    expected_results = {
        "ROUGE-1": "42.5-44.0",
        "ROUGE-2": "21.0-23.0", 
        "ROUGE-L": "42.0-44.5",
        "BERTScore": "0.92-0.94",
        "vs_Qwen72B_direct": "+3.5-5.0 ROUGE-L"
    }
    
    logger.success("EXPECTED RESULTS:")
    for metric, score in expected_results.items():
        logger.success(f"   {metric}: {score}")
    
    wandb.finish()

if __name__ == "__main__":
    main()