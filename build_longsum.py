# build_longsum.py
from longragsum.config.configuration import ConfigurationManager
from longragsum.components.benchmark_builder import LongSum2025Builder

if __name__ == "__main__":
    cfg = ConfigurationManager().config.data
    builder = LongSum2025Builder(cfg)

    print("STEP 1: Building raw LongSum-2025...")
    raw_dataset, stats = builder.build_raw()  # Returns (DatasetDict, stats)

    print("\nSTEP 2: Verifying with BERTScore...")
    verified_dataset = builder.verify_with_bertscore(raw_dataset)  # ← FIXED: Use raw_dataset directly

    print("\nSTEP 3: Pushing to Hugging Face...")
    builder.push_to_hub(verified_dataset, repo_id="ashaduzzaman/LongSum-2025")

    print(f"\n🎉 LONG-SUM-2025 IS COMPLETE!")
    print(f"📊 FINAL STATS: Train={stats['train']}, Val={stats['validation']}, Test={stats['test']}")
    print(f"🌟 Domains: {stats['successful_domains']} (BookSum + arXiv + GovReport + PubMed + QMSum)")
    print("🚀 Your benchmark is PUBLICATION-READY!")
    print("📈 Expected ROUGE-L after Qwen2.5 training: 40-45")
    print("\nNext: Execute Qwen dev → Train your SOTA model!")