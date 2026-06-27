from embeddings.dense_embeddings import DenseEmbedder
from vectorstore.chroma_store import ChromaStore


class DenseRetriever:

    def __init__(self):

        self.embedder = DenseEmbedder()
        self.vector_store = ChromaStore()

    def retrieve(self, query: str, top_k: int = 5):

        query_embedding = self.embedder.embed_query(query)

        return self.vector_store.search(query_embedding, top_k)
