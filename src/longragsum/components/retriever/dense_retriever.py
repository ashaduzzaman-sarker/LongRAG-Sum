# src/longragsum/components/retriever/dense_retriever.py
import faiss
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from longragsum.logging.logger import logger
from tqdm.auto import tqdm

class DenseRetriever:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading retriever model: {config.model_name} → {self.device}")
        self.model = SentenceTransformer(config.model_name, device=self.device)
        self.model.eval()

        self.index = None
        self.chunks_df = None
        self.metadata_df = None
        self.dimension = None

    def build_index(self, force_rebuild=False):
        index_path = self.config.index_dir / "faiss.index"
        chunks_path = Path("artifacts/data/processed/chunks.parquet")
        meta_path = Path("artifacts/data/processed/metadata.parquet")

        if index_path.exists() and not force_rebuild:
            logger.info(f"Loading existing index from {index_path}")
            self.index = faiss.read_index(str(index_path))
            self.chunks_df = pd.read_parquet(chunks_path)
            self.metadata_df = pd.read_parquet(meta_path)
            self.dimension = self.index.d
            logger.success(f"Loaded FAISS index with {self.index.ntotal:,} passages")
            return

        # Load chunks
        self.chunks_df = pd.read_parquet(chunks_path)
        self.metadata_df = pd.read_parquet(meta_path)
        texts = self.chunks_df["text"].tolist()

        logger.info(f"Encoding {len(texts):,} chunks with {self.config.model_name}")
        embeddings = []
        batch_size = self.config.batch_size

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                batch_emb = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            embeddings.append(batch_emb)

        embeddings = np.vstack(embeddings).astype("float32")
        self.dimension = embeddings.shape[1]

        # Build FAISS index (Inner Product = cosine after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        # Save everything
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        logger.success(f"FAISS index built & saved → {self.index.ntotal:,} passages")

    @torch.no_grad()
    def retrieve(self, query: str, k: int | None = None):
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        k = k or self.config.top_k
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        scores, indices = self.index.search(query_emb, k)
        scores = scores.flatten()
        indices = indices.flatten()

        results = []
        for score, idx in zip(scores, indices):
            chunk_text = self.chunks_df.iloc[idx]["text"]
            meta = self.metadata_df.iloc[idx].to_dict()
            meta.update({"score": float(score), "rank": len(results)+1})
            results.append({"text": chunk_text, "metadata": meta})

        return results