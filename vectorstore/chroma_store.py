import chromadb
from dataclasses import asdict
from models.document import Document


class ChromaStore:

    def __init__(self, path: str = "./database", collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=path)

        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: list[Document], embeddings: list[list[float]]):

        self.collection.add(
            ids=[str(doc.metadata.chunk_id) for doc in documents],
            documents=[doc.content for doc in documents],
            embeddings=embeddings,
            metadatas=[asdict(doc.metadata) for doc in documents],
        )
