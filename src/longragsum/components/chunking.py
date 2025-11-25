import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
from longragsum.logging.logger import logger
from tqdm import tqdm

class DocumentChunker:
    def __init__(self, config):
        self.config = config

    def chunk_text(self, text: str):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence.split())
            if current_length + sentence_len > self.config.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Overlap
                overlap = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s.split()) <= self.config.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_length += len(s.split())
                    else:
                        break
                current_chunk = overlap + [sentence]
                current_length = overlap_length + sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return [c for c in chunks if len(c.split()) >= 50]

    def process_dataset(self, parquet_path):
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        chunks = []
        metadata = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
            doc_chunks = self.chunker.chunk_text(row["chapter"])
            for i, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                metadata.append({
                    "source_id": idx,
                    "book_title": row.get("book_title", row.get("title", "")),
                    "summary": row["summary"],
                    "chunk_id": i,
                    "total_chunks": len(doc_chunks)
                })

        save_dir = Path("artifacts/data/processed")
        save_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": chunks}).to_parquet(save_dir / "chunks.parquet")
        pd.DataFrame(metadata).to_parquet(save_dir / "metadata.parquet")
        logger.success(f"Chunking complete → {len(chunks)} chunks saved")