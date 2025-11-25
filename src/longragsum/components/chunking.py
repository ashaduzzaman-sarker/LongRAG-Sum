# src/longragsum/components/chunking.py
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize
from longragsum.logging.logger import logger
from tqdm import tqdm
import pandas as pd
from pathlib import Path

class DocumentChunker:
    def __init__(self, config):
        self.config = config

    def chunk_text(self, text: str):
        sentences = sent_tokenize(text.strip())
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence.split())
            if current_length + sentence_len > self.config.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Create overlap
                overlap = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_len + s_len <= self.config.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_len += s_len
                    else:
                        break
                current_chunk = overlap + [sentence]
                current_length = overlap_len + sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c.strip() for c in chunks if len(c.split()) >= 50]

    def process_dataset(self, parquet_path):
        df = pd.read_parquet(parquet_path)

        chunks = []
        metadata = []

        save_dir = Path("artifacts/data/processed")
        save_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking GovReport documents"):
            # GovReport uses "report" field
            text = row["report"]
            if not text or len(text.split()) < 1000:
                continue

            doc_chunks = self.chunk_text(text)

            for i, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                metadata.append({
                    "source_id": idx,
                    "doc_id": row.get("id", f"doc_{idx}"),
                    "title": f"GovReport_{row.get('id', idx)}",  # fallback title
                    "summary": row["summary"],
                    "chunk_id": i,
                    "total_chunks": len(doc_chunks)
                })

        # Save
        pd.DataFrame({"text": chunks}).to_parquet(save_dir / "chunks.parquet")
        pd.DataFrame(metadata).to_parquet(save_dir / "metadata.parquet")

        logger.success(f"Chunking complete! → {len(chunks):,} chunks from {len(df)} documents saved")
        logger.info(f"   → Average chunks per doc: {len(chunks)/len(df):.1f}")