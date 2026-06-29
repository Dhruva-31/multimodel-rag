from embeddings.dense_embeddings import DenseEmbedder
from vectorstore.chroma_store import ChromaStore


class DenseRetriever:

    def __init__(self):
        self.embedder = DenseEmbedder()
        self.vector_store = ChromaStore()
        self._cache: dict[str, list[float]] = {}

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self._get_embedding(query)
        return self.vector_store.search(query_embedding, top_k)

    def _get_embedding(self, query: str) -> list[float]:
        normalized = query.strip().lower()
        if normalized not in self._cache:
            self._cache[normalized] = self.embedder.embed_query(query)
        return self._cache[normalized]

    def clear_cache(self):
        self._cache.clear()
