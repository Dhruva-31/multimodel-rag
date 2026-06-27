from ingestion.pdf_loader import PDFLoader
from preprocessing.chunker import Chunker
from embeddings.dense_embeddings import DenseEmbedder
from vectorstore.chroma_store import ChromaStore


def main():

    print("Loading PDF...")
    loader = PDFLoader()

    documents = loader.load("uploads/sample.pdf")

    print(f"Loaded {len(documents)} pages")

    print("\nChunking...")
    chunker = Chunker()

    chunked_documents = chunker.chunk(documents)

    print(f"Created {len(chunked_documents)} chunks")

    print("\nGenerating embeddings...")
    embedder = DenseEmbedder()

    embeddings = embedder.embed_documents(
        [document.content for document in chunked_documents]
    )

    print(f"Generated {len(embeddings)} embeddings")

    print(f"Embedding dimension: {len(embeddings[0])}")

    print("\nStoring in ChromaDB...")
    vector_store = ChromaStore()

    vector_store.add_documents(chunked_documents, embeddings)

    print(f"Total vectors stored: " f"{vector_store.collection.count()}")

    print("\nSample chunk:")
    print(chunked_documents[0].metadata)

    print(chunked_documents[0].content[:300])


if __name__ == "__main__":
    main()
