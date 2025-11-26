# test_rag_full.py
from longragsum.config.configuration import ConfigurationManager
from longragsum.components.retriever.dense_retriever import DenseRetriever
from longragsum.components.reader.rag_reader import RAGReader

if __name__ == "__main__":
    cfg = ConfigurationManager()
    retriever_cfg = cfg.get_retriever_config()
    reader_cfg = cfg.config.reader  # simple dict is fine here

    # Load retriever + index
    retriever = DenseRetriever(retriever_cfg)
    retriever.build_index(force_rebuild=False)

    # Load RAG reader
    reader = RAGReader(retriever, reader_cfg)

    # REAL TEST
    query = "What are the key findings and policy recommendations regarding renewable energy adoption in the United States?"

    print("Generating RAG summary...\n")
    result = reader.summarize(query, k=32, max_new_tokens=600)

    print("="*80)
    print("FINAL RAG SUMMARY")
    print("="*80)
    print(result["summary"])
    print("\nUsed passages from documents:", set(result["retrieved_docs"]))