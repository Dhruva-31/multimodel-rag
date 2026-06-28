from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = ("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[
            tuple[
                str,
                dict,
            ]
        ],
        top_k: int = 5,
    ) -> list[
        tuple[
            str,
            float,
            dict,
        ]
    ]:

        pairs = [
            (
                query,
                document,
            )
            for document, _ in candidates
        ]

        scores = [float(score) for score in self.model.predict(pairs)]

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            (
                document,
                score,
                metadata,
            )
            for (
                document,
                metadata,
            ), score in ranked[:top_k]
        ]
