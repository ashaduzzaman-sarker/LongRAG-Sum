# build_longsum.py
from longragsum.config.configuration import ConfigurationManager
from longragsum.components.benchmark_builder import LongSum2025Builder

if __name__ == "__main__":
    cfg = ConfigurationManager().config.data
    builder = LongSum2025Builder(cfg)

    print("STEP 1: Building raw LongSum-2025...")
    raw_dataset = builder.build_raw()

    print("\nSTEP 2: Verifying with BERTScore...")
    verified_dataset = builder.verify_with_bertscore(raw_dataset)

    print("\nSTEP 3: Pushing to Hugging Face...")
    builder.push_to_hub(verified_dataset, repo_id="yourusername/LongSum-2025")

    print("\nLONG-SUM-2025 IS COMPLETE!")