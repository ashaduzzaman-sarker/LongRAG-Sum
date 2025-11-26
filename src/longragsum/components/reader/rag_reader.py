# src/longragsum/components/reader/rag_reader.py
import unsloth
import torch
from transformers import AutoTokenizer, GenerationConfig
from peft import PeftModel
from longragsum.logging.logger import logger
from tqdm import tqdm

class RAGReader:
    def __init__(self, retriever, config):
        self.retriever = retriever
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading {config.base_model} with 128k context (official Meta weights)")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            use_fast=True,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, query: str, retrieved_passages, max_length=32000):
        system = "You are an expert summarizer. Write a concise, accurate summary using ONLY the provided passages."
        passages_text = "\n\n".join([
            f"Passage {i+1}: {doc['text']}" 
            for i, doc in enumerate(retrieved_passages)
        ])
        user = f"""Question: {query}

Relevant passages:
{passages_text}

Provide a clear, comprehensive summary in 3–6 paragraphs. Cite passage numbers when used."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    @torch.no_grad()
    def summarize(self, query: str, k: int | None = None, max_new_tokens=512):
        retrieved = self.retriever.retrieve(query, k=k or self.retriever.config.top_k)
        prompt = self.build_prompt(query, retrieved)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length
        ).to(self.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = response.split("assistant")[-1].strip()
        return {
            "summary": summary,
            "num_passages_used": len(retrieved),
            "retrieved_docs": [r["metadata"]["doc_id"] for r in retrieved]
        }