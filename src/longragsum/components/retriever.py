# src/longragsum/components/retriever.py
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from pathlib import Path
from longragsum.logging.logger import logger

class BGERetriever:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.doc_embeddings = None
        self.doc_ids = None
        self._setup_model()
    
    def _setup_model(self):
        """Initialize BGE-m3 retriever"""
        logger.info(f"Loading BGE-m3 retriever: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        logger.success("BGE-m3 retriever loaded")
    
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build FAISS index for documents"""
        logger.info(f"Building index for {len(documents)} documents...")
        
        # Generate embeddings
        self.doc_embeddings = self.model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # FAISS Index
        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity
        self.index.add(self.doc_embeddings.astype('float32'))
        
        self.doc_ids = doc_ids
        logger.success(f"Index built: {len(documents)} docs, {dimension}d embeddings")
    
    def retrieve(self, query: str, top_k: int = 8) -> List[Tuple[str, float]]:
        """Retrieve top-k relevant chunks"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(score)))
        
        return results
    
    def hybrid_search(self, query: str, dense_top_k: int = 8, sparse_weight: float = 0.3) -> List[Tuple[str, float]]:
        """Hybrid dense + sparse search (BGE-m3 optimized)"""
        dense_results = self.retrieve(query, dense_top_k)
        
        # BM25-style sparse scoring (simple TF-IDF approximation)
        sparse_scores = self._bm25_approx(query, self.doc_ids)
        
        # Combine scores
        combined = {}
        for doc_id, dense_score in dense_results:
            combined[doc_id] = dense_score
        
        for doc_id, sparse_score in sparse_scores:
            if doc_id in combined:
                combined[doc_id] = (1-sparse_weight) * combined[doc_id] + sparse_weight * sparse_score
            else:
                combined[doc_id] = sparse_score
        
        # Sort and return top-k
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:dense_top_k]
    
    def _bm25_approx(self, query: str, doc_ids: List[str]) -> List[Tuple[str, float]]:
        """Simple BM25 approximation for sparse retrieval"""
        query_words = query.lower().split()
        scores = {}
        
        for doc_id in doc_ids:
            score = sum(1 for word in query_words if word in doc_id.lower())
            scores[doc_id] = score / max(1, len(query_words))
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]