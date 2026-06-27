from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ):

        pairs = [(query, document) for document in documents]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]
