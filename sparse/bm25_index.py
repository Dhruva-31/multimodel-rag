from rank_bm25 import BM25Okapi

from models.document import Document


class BM25Retriever:

    def __init__(self):
        self.documents = []
        self.bm25 = None

    def index(self, documents: list[Document]):

        self.documents = documents

        tokenized_docs = [doc.content.lower().split() for doc in documents]

        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, top_k: int = 5):

        scores = self.bm25.get_scores(query.lower().split())

        scored = list(zip(self.documents, scores))

        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]
