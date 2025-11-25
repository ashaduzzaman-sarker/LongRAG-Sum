# test_ingestion.py
from longragsum.config.configuration import ConfigurationManager
from longragsum.components.data_ingestion import LongFormDataIngestion
from longragsum.components.chunking import DocumentChunker

if __name__ == "__main__":
    cfg = ConfigurationManager()
    ingest_cfg = cfg.get_data_ingestion_config()
    chunk_cfg = cfg.get_chunking_config()

    # Step 1: Download & filter long documents
    ingestion = LongFormDataIngestion(ingest_cfg)
    parquet_path = ingestion.download_and_save()

    # Step 2: Chunk them
    chunker = DocumentChunker(chunk_cfg)
    chunker.process_dataset(parquet_path)