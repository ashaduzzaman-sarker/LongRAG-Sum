from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_dir: Path
    dataset_name: str
    dataset_config: str
    split: str
    max_samples: int | None

@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    min_chunk_length: int = 50

@dataclass(frozen=True)
class RetrieverConfig:
    model_name: str
    top_k: int
    batch_size: int
    device: str
    index_dir: Path

@dataclass(frozen=True)
class ReaderConfig:
    base_model: str
    max_seq_length: int
    use_4bit: bool
    lora_r: int
    lora_alpha: int