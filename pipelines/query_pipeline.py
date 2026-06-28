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


def ask_question(
    query: str,
    bm25,
):

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=bm25,
    )

    hybrid_results = hybrid.retrieve(
        query,
        top_k=20,
    )

    candidate_docs = [
        (
            text,
            metadata,
        )
        for _, _, text, metadata in hybrid_results
    ]

    reranked = reranker.rerank(
        query,
        candidate_docs,
        top_k=10,
    )

    context = context_builder.build(
        query,
        reranked,
    )

    response = generator.generate(context)

    return response
