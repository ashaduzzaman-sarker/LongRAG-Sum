# src/longragsum/components/rag_pipeline.py
from typing import List, Dict, Any
from .reader import QwenReader
from .retriever import BGERetriever
from datasets import Dataset
import torch
from longragsum.logging.logger import logger

class LongRAGPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reader = QwenReader(config['model']['dev_reader'])
        self.retriever = BGERetriever()
        self.max_chunks = config['data']['longsum_2025']['chunk_size']
        self.top_k = 8
        
    def preprocess_documents(self, dataset: Dataset) -> List[str]:
        """Chunk long documents for retrieval"""
        logger.info("Preprocessing documents for chunking...")
        
        chunks = []
        chunk_ids = []
        
        for i, example in enumerate(dataset):
            text = example['text']
            
            # Simple chunking (improve with semantic chunking later)
            words = text.split()
            for j in range(0, len(words), self.max_chunks // 4):  # ~256 word chunks
                chunk = ' '.join(words[j:j + self.max_chunks // 4])
                if len(chunk.split()) >= 50:  # Minimum chunk size
                    chunks.append(chunk)
                    chunk_ids.append(f"{example['id']}_chunk_{j//(self.max_chunks//4)}")
        
        logger.success(f"Created {len(chunks)} chunks from {len(dataset)} documents")
        return chunks, chunk_ids
    
    def build_retrieval_index(self, chunks: List[str], chunk_ids: List[str]):
        """Build retriever index"""
        self.retriever.build_index(chunks, chunk_ids)
        logger.success("Retrieval index built")
    
    def generate_rag_summary(self, document_id: str, full_text: str, query: str = "Summarize:") -> str:
        """Generate RAG summary for a document"""
        
        # Retrieve relevant chunks
        retrieved = self.retriever.hybrid_search(query, top_k=self.top_k)
        context_chunks = [chunk for chunk_id, score in retrieved[:self.top_k] 
                         for chunk in [self._get_chunk_by_id(chunk_id)] if chunk]
        
        if not context_chunks:
            logger.warning(f"No chunks retrieved for {document_id}")
            return ""
        
        # Generate with Qwen
        summary = self.reader.generate_summary(
            context_chunks,
            query=query,
            max_new_tokens=512,
            temperature=0.7
        )
        
        return summary
    
    def _get_chunk_by_id(self, chunk_id: str) -> str:
        """Get chunk by ID (implement chunk storage)"""
        # TODO: Implement proper chunk storage/retrieval
        # For now, return dummy
        return "Retrieved chunk content"
    
    def batch_evaluate(self, test_dataset: Dataset) -> Dict[str, Any]:
        """Batch evaluation on test set"""
        logger.info(f"Starting batch evaluation on {len(test_dataset)} test examples...")
        
        results = {
            'generated_summaries': [],
            'document_ids': [],
            'queries': []
        }
        
        for example in test_dataset:
            doc_id = example['id']
            full_text = example['text']
            reference_summary = example['summary']
            
            # Generate RAG summary
            query = f"Summarize the following {example['domain']} document:"
            generated_summary = self.generate_rag_summary(
                doc_id, full_text, query
            )
            
            results['generated_summaries'].append(generated_summary)
            results['document_ids'].append(doc_id)
            results['queries'].append(query)
        
        logger.success(f"Generated {len(results['generated_summaries'])} summaries")
        return results