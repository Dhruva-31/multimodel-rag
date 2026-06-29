import time
from guardrails.input_guardrail import check as input_check
from guardrails.output_guardrail import check as output_check
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

from generation.context_builder import ContextBuilder
from generation.generator import Generator

dense_retriever = DenseRetriever()
reranker = CrossEncoderReranker()
generator = Generator()
context_builder = ContextBuilder()


def get_retrieval_config(query: str):

    q = query.lower().strip()

    # =====================================================
    # COMPARISON
    # =====================================================

    if (
        q.startswith(("compare",))
        or " difference " in f" {q} "
        or " versus " in f" {q} "
        or " vs " in f" {q} "
    ):
        return {
            "retrieve_k": 40,
            "candidate_k": 20,
            "rerank_k": 15,
            "use_reranker": True,
            "query_type": "compare",
            "max_chars": 8000,
        }

    # =====================================================
    # FILTER / ENUMERATION
    # =====================================================

    if (
        q.startswith(
            (
                "list",
                "filter",
                "find",
                "extract",
                "show",
            )
        )
        or "list all" in q
        or "find all" in q
        or "show all" in q
        or "filter out" in q
    ):
        return {
            "retrieve_k": 50,
            "candidate_k": 50,
            "rerank_k": 0,
            "use_reranker": False,
            "query_type": "filter",
            "max_chars": 10000,
        }

    # =====================================================
    # SUMMARY
    # =====================================================

    if (
        q.startswith(("summarize", "summarise"))
        or "summary" in q
        or "overview" in q
        or "summarize the entire" in q
        or "summarise the entire" in q
        or "summarize the document" in q
        or "summarize this pdf" in q
        or "whole document" in q
        or "entire document" in q
    ):
        return {
            "retrieve_k": 50,
            "candidate_k": 30,
            "rerank_k": 20,
            "use_reranker": True,
            "query_type": "summary",
            "max_chars": 12000,
        }

    # =====================================================
    # ANALYTICAL
    # =====================================================

    if any(
        word in q
        for word in [
            "explain",
            "why",
            "how",
            "analyze",
            "analyse",
            "describe",
        ]
    ):
        return {
            "retrieve_k": 40,
            "candidate_k": 20,
            "rerank_k": 15,
            "use_reranker": True,
            "query_type": "explain",
            "max_chars": 8000,
        }

    # =====================================================
    # FACT LOOKUP
    # =====================================================

    return {
        "retrieve_k": 20,
        "candidate_k": 10,
        "rerank_k": 10,
        "use_reranker": True,
        "query_type": "default",
        "max_chars": 5000,
    }


def build_retriever(bm25) -> HybridRetriever:
    dense_retriever.clear_cache()
    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=bm25,
    )


def ask_question(query: str, retriever: HybridRetriever):

    passed, message = input_check(query)
    if not passed:
        return message

    t0 = time.perf_counter()

    config = get_retrieval_config(query)

    hybrid_results = retriever.retrieve(
        query,
        top_k=config["retrieve_k"],
    )

    t1 = time.perf_counter()

    candidate_docs = [
        (text, metadata)
        for _, _, text, metadata in hybrid_results[: config["candidate_k"]]
    ]

    if config["use_reranker"]:

        reranked = reranker.rerank(
            query,
            candidate_docs,
            top_k=config["rerank_k"],
        )

    else:

        reranked = [
            (
                text,
                score,
                metadata,
            )
            for _, score, text, metadata in hybrid_results[: config["candidate_k"]]
        ]
    t2 = time.perf_counter()

    context = context_builder.build(query, reranked, max_chars=config["max_chars"])
    response = generator.generate(context)
    t3 = time.perf_counter()

    response = output_check(response, reranked)

    print("\n========= QUERY TIMING =========")
    print(f"Retrieval (hybrid):  {(t1 - t0) * 1000:.1f}ms")
    print(f"Reranking:           {(t2 - t1) * 1000:.1f}ms")
    print(f"Generation:          {(t3 - t2) * 1000:.1f}ms")
    print(f"Total:               {(t3 - t0) * 1000:.1f}ms")
    print("=================================\n")

    return response
