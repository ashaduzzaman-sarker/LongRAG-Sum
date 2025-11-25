from longragsum.utils.common import read_yaml, create_directories
from longragsum.entity.config_entity import (
    DataIngestionConfig, ChunkingConfig, RetrieverConfig, ReaderConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath="config/config.yaml"):
        self.config = read_yaml(Path(config_filepath))
        create_directories([self.config.artifacts.ARTIFACTS_DIR])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data
        artifacts = self.config.artifacts
        create_directories([artifacts.DATA_DIR])
        return DataIngestionConfig(
            raw_data_dir=Path(artifacts.DATA_DIR) / "raw",
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.get("dataset_config", "default"),
            split=cfg.dataset_split,
            max_samples=cfg.get("max_samples")
        )

    def get_chunking_config(self) -> ChunkingConfig:
        c = self.config.data
        return ChunkingConfig(
            chunk_size=c.chunk_size,
            chunk_overlap=c.chunk_overlap
        )

    def get_retriever_config(self) -> RetrieverConfig:
        r = self.config.retriever
        a = self.config.artifacts
        create_directories([a.INDEX_DIR, a.RETRIEVER_DIR])
        return RetrieverConfig(
            model_name=r.model_name,
            top_k=self.config.data.top_k,
            batch_size=r.batch_size,
            device=r.device,
            index_dir=Path(a.INDEX_DIR)
        )