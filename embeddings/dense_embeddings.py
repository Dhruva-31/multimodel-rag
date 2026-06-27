from fastembed import TextEmbedding


class DenseEmbedder:

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        embeddings = list(self.model.embed(texts))

        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> list[float]:

        embedding = next(iter(self.model.embed([query])))

        return embedding.tolist()
