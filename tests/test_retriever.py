# test_retriever.py
from longragsum.config.configuration import ConfigurationManager
from longragsum.components.retriever.dense_retriever import DenseRetriever

if __name__ == "__main__":
    cfg = ConfigurationManager()
    retriever_cfg = cfg.get_retriever_config()
    retriever = DenseRetriever(retriever_cfg)

    # Build (or load) the index
    retriever.build_index(force_rebuild=False)  # Set True only first time

    # Test retrieval!
    query = "What are the main recommendations on climate change mitigation?"
    results = retriever.retrieve(query, k=8)

    print(f"\nQuery: {query}\n")
    for r in results:
        meta = r["metadata"]
        print(f"[{r['metadata']['rank']}] Score: {meta['score']:.4f} | Doc: {meta['doc_id']}")
        print(f"    → {r['text'][:200]}...\n")