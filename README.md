# LongRAG-Sum
**Retrieval-Augmented Long-Form Summarization**  
Target: ACL/EMNLP 2026  

State-of-the-art summarization for books, scientific papers, and legal documents (>100K tokens)  
Using open LLMs + dense retrieval + LoRA fine-tuning.

Currently under active development for publication.

LongRAG-Sum/
├── config/
│   └── config.yaml                 # ← Qwen2.5 + LongSum-2025 config
├── src/longragsum/
│   ├── components/
│   │   ├── reader.py              # ← Qwen2.5-7B/72B
│   │   ├── retriever.py           # ← BGE-m3 upgrade
│   │   ├── rag_pipeline.py        # ← LongSum-2025 optimized
│   │   ├── evaluator.py           # ← ROUGE/AlignScore
│   │   └── __init__.py
│   ├── training/
│   │   ├── train_rag_lora.py      # ← Qwen LoRA training
│   │   └──__init__.py
│   ├── evaluation/
│   │   ├── generate_summaries.py  # ← Batch evaluation
│   │   └──__init__.py
├── scripts/
│   ├── train_qwen.py             # ← Main training script
│   ├── evaluate_longsum.py       # ← Full evaluation
│   └── benchmark_results.py      # ← Paper tables
├── artifacts/
│   └── longsum_2025/             # ✅ Already built
└── requirements.txt              # ← Updated dependencies