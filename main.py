from ingestion.pdf_loader import PDFLoader
from preprocessing.chunker import Chunker

from sparse.bm25_index import BM25Retriever

from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

from generation.context_builder import ContextBuilder
from generation.generator import Generator

QUERY = "What AI engineering projects help get jobs?"


def main():

    # =====================================
    # Build BM25
    # (temporary until persistence)
    # =====================================

    print("Building BM25...")

    loader = PDFLoader()
    chunker = Chunker()

    documents = loader.load("uploads/sample.pdf")

    chunked_documents = chunker.chunk(documents)

    bm25 = BM25Retriever()

    bm25.index(chunked_documents)

    # =====================================
    # Dense Retriever
    # =====================================

    dense = DenseRetriever()

    # =====================================
    # Hybrid Retrieval
    # =====================================

    print("\nHybrid Retrieval...")

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=bm25,
    )

    hybrid_results = hybrid.retrieve(
        QUERY,
        top_k=10,
    )

    # =====================================
    # Cross Encoder
    # =====================================

    print("\nReranking...")

    reranker = CrossEncoderReranker()

    candidate_docs = [text for _, _, text in hybrid_results]

    reranked = reranker.rerank(
        QUERY,
        candidate_docs,
        top_k=5,
    )

    print("\nTop Documents:")

    for rank, (doc, score) in enumerate(
        reranked,
        start=1,
    ):
        print()
        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(doc[:200])

    # =====================================
    # Context Building
    # =====================================

    print("\nBuilding Context...")

    builder = ContextBuilder()

    context = builder.build(
        QUERY,
        [doc for doc, _ in reranked],
    )

    # =====================================
    # LLM
    # =====================================

    print("\nGenerating Answer...")

    generator = Generator(model="qwen3:4b")

    answer = generator.generate(context)

    print("\n====================")
    print("FINAL ANSWER")
    print("====================\n")

    print(answer)


if __name__ == "__main__":
    main()
