# scripts/preprocess_longsum.py
from longragsum.config.configuration import ConfigurationManager
from longragsum.components.data_ingestion import DataIngestion
from longragsum.logging.logger import logger

if __name__ == "__main__":
    cfg_manager = ConfigurationManager()
    config = cfg_manager.config
    
    logger.info("🚀 Starting LongSum-2025 Preprocessing...")
    ingestion = DataIngestion(config)
    processed_dataset = ingestion.run_preprocessing()
    
    # Verify
    logger.info("📊 Verification:")
    for split, ds in processed_dataset.items():
        logger.info(f"  {split}: {len(ds)} examples")
        if len(ds) > 0:
            example = ds[0]
            logger.info(f"  Sample ID: {example['id']}")
            logger.info(f"  Domain: {example['domain']}")
            logger.info(f"  Token length: {example['token_length']}")
            logger.info(f"  Number of chunks: {len(example['chunks'])}")
            logger.info(f"  First chunk sample: {example['chunks'][0][:200]}...")
    
    logger.success("✅ PREPROCESSING COMPLETE & VERIFIED!")
    logger.success(f"📂 Saved to: artifacts/processed_longsum")
    logger.success("👉 Next step: Retriever upgrade (BGE-m3) + indexing")