# train_rag_lora.py
import os
import torch
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from longragsum.config.configuration import ConfigurationManager
from longragsum.components.retriever.dense_retriever import DenseRetriever
from tqdm import tqdm

# Clear cache
torch.cuda.empty_cache()

# ========================
# 1. Load Configs & Retriever
# ========================
print("Loading configuration and retriever...")
cfg = ConfigurationManager()
retriever_cfg = cfg.get_retriever_config()
retriever = DenseRetriever(retriever_cfg)
retriever.build_index(force_rebuild=False)

# Load tokenizer once (we'll reuse it)
tokenizer = retriever.model.tokenizer  # BGE tokenizer is fine, but we need Llama's
print("Loading Llama-3.1-8B-Instruct tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_fast=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# ========================
# 2. Load GovReport validation set (our training data)
# ========================
print("Loading GovReport validation split...")
raw_dataset = load_dataset("ccdv/govreport-summarization", split="validation[:400]")  # 400 = safe for T4

# ========================
# 3. Build RAG-formatted training examples (top-8 passages)
# ========================
def format_rag_example(example):
    query = "Summarize the following government report."
    retrieved = retriever.retrieve(query, k=8)  # top-8 only → fits T4

    passages = "\n\n".join([
        f"Passage {i+1}: {doc['text']}" for i, doc in enumerate(retrieved)
    ])

    messages = [
        {"role": "system", "content": "You are an expert government report summarizer. Use only the provided passages."},
        {"role": "user", "content": f"{query}\n\nRelevant passages:\n{passages}\n\nWrite a comprehensive summary."},
        {"role": "assistant", "content": example["summary"]}
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted}

print("Formatting training examples with retrieved passages...")
train_data = []
for ex in tqdm(raw_dataset, desc="Formatting"):
    try:
        train_data.append(format_rag_example(ex))
    except:
        continue  # skip broken

train_dataset = Dataset.from_list(train_data)
print(f"Final training examples: {len(train_dataset)}")

# ========================
# 4. Load 4-bit Llama-3.1-8B (official 128k version)
# ========================
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

print("Loading Llama-3.1-8B-Instruct in 4-bit...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# ========================
# 5. Apply QLoRA
# ========================
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========================
# 6. Training Arguments (T4-optimized)
# ========================
training_args = TrainingArguments(
    output_dir="artifacts/lora_checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to=[],
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    run_name="LongRAG-Sum-GovReport-v1"
)

# ========================
# 7. Start Training
# ========================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=8192,
    packing=False,
)

print("STARTING QLoRA FINE-TUNING...")
trainer.train()

# ========================
# 8. Save Final Adapter
# ========================
final_path = "artifacts/lora_adapter_final"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print(f"""
FINE-TUNING COMPLETE!
Your LongRAG-Sum adapter is saved at: {final_path}
Next steps:
1. Run generate_test_summaries.py
2. Run evaluate.py
3. Submit to ACL 2026
""")