from collections import defaultdict
from retrieval.dense_retriever import DenseRetriever
from sparse.bm25_index import BM25Retriever
from typing import cast


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

        metadatas = {}

        if (
            dense_results["documents"] is None
            or dense_results["metadatas"] is None
            or sparse_results is None
        ):
            return []

        for rank, metadata in enumerate(dense_results["metadatas"][0]):

            chunk_id = metadata["chunk_id"]

            scores[chunk_id] += 1 / (rrf_k + rank)

            documents[chunk_id] = dense_results["documents"][0][rank]

            metadatas[chunk_id] = {
                "source": metadata["source"],
                "filename": metadata["filename"],
                "type": metadata["type"],
                "page": metadata.get("page"),
                "paragraph": metadata.get("paragraph"),
                "chunk_id": metadata.get("chunk_id"),
                "chunk_size": metadata.get("chunk_size"),
            }

        for rank, (doc, _) in enumerate(sparse_results):

            chunk_id = doc.metadata.chunk_id

            scores[chunk_id] += 1 / (rrf_k + rank)

            documents[chunk_id] = doc.content

            if chunk_id not in metadatas:

                metadatas[chunk_id] = {
                    "source": doc.metadata.source,
                    "filename": doc.metadata.filename,
                    "type": doc.metadata.type,
                    "page": doc.metadata.page,
                    "paragraph": doc.metadata.paragraph,
                    "chunk_id": doc.metadata.chunk_id,
                    "chunk_size": doc.metadata.chunk_size,
                }

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
                metadatas[chunk_id],
            )
            for chunk_id, score in ranked[:top_k]
        ]
