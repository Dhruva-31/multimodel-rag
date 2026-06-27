from collections import defaultdict

from retrieval.dense_retriever import DenseRetriever
from sparse.bm25_index import BM25Retriever


class HybridRetriever:

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
    ):

        dense_results = self.dense.retrieve(
            query,
            top_k=top_k,
        )

        sparse_results = self.sparse.retrieve(
            query,
            top_k=top_k,
        )

        scores = defaultdict(float)
        documents = {}

        # Dense
        for rank, metadata in enumerate(dense_results["metadatas"][0]):

            chunk_id = metadata["chunk_id"]

            scores[chunk_id] += 1 / (rrf_k + rank)

            documents[chunk_id] = dense_results["documents"][0][rank]

        # Sparse
        for rank, (doc, _) in enumerate(sparse_results):

            chunk_id = doc.metadata.chunk_id

            scores[chunk_id] += 1 / (rrf_k + rank)

            documents[chunk_id] = doc.content

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            (
                chunk_id,
                score,
                documents[chunk_id],
            )
            for chunk_id, score in ranked[:top_k]
        ]
