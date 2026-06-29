import time
import pipelines.ingestion_pipeline as ingestion

from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

from generation.context_builder import ContextBuilder
from generation.generator import Generator

dense_retriever = DenseRetriever()
reranker = CrossEncoderReranker()
generator = Generator()
context_builder = ContextBuilder()


def build_retriever(bm25) -> HybridRetriever:
    dense_retriever.clear_cache()
    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=bm25,
    )


def ask_question(query: str, retriever: HybridRetriever):

    t0 = time.perf_counter()

    hybrid_results = retriever.retrieve(query, top_k=20)
    t1 = time.perf_counter()

    candidate_docs = [(text, metadata) for _, _, text, metadata in hybrid_results[:10]]

    reranked = reranker.rerank(query, candidate_docs, top_k=10)
    t2 = time.perf_counter()

    context = context_builder.build(query, reranked)
    response = generator.generate(context)
    t3 = time.perf_counter()

    print("\n========= QUERY TIMING =========")
    print(f"Retrieval (hybrid):  {(t1 - t0) * 1000:.1f}ms")
    print(f"Reranking:           {(t2 - t1) * 1000:.1f}ms")
    print(f"Generation:          {(t3 - t2) * 1000:.1f}ms")
    print(f"Total:               {(t3 - t0) * 1000:.1f}ms")
    print("=================================\n")

    return response
