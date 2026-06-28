from typing import cast
import chromadb
from dataclasses import asdict
from models.document import Document
from chromadb.api.types import Embedding
import uuid


class ChromaStore:

    def __init__(self, path: str = "./database"):
        self.client = chromadb.PersistentClient(path=path)

        self.collection_name = "session_" + str(uuid.uuid4())

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_documents(self, documents: list[Document], embeddings: list[list[float]]):

        metadatas = []

        for doc in documents:

            metadata = {k: v for k, v in asdict(doc.metadata).items() if v is not None}

            metadatas.append(metadata)

        self.collection.add(
            ids=[str(doc.metadata.chunk_id) for doc in documents],
            documents=[doc.content for doc in documents],
            embeddings=cast(
                list[Embedding],
                embeddings,
            ),
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_k: int = 5):

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def delete_collection(self):

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
