from ingestion.pdf_loader import PDFLoader
from preprocessing.chunker import Chunker
from embeddings.dense_embeddings import DenseEmbedder
from vectorstore.chroma_store import ChromaStore

from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever

from sparse.bm25_index import BM25Retriever


def main():

    # =====================================
    # INGESTION
    # =====================================
    print("Loading PDF...")

    loader = PDFLoader()

    documents = loader.load("uploads/sample.pdf")

    print(f"Loaded {len(documents)} pages")

    # =====================================
    # CHUNKING
    # =====================================
    print("\nChunking...")

    chunker = Chunker()

    chunked_documents = chunker.chunk(documents)

    print(f"Created {len(chunked_documents)} chunks")

    # =====================================
    # EMBEDDINGS
    # =====================================
    print("\nGenerating embeddings...")

    embedder = DenseEmbedder()

    embeddings = embedder.embed_documents(
        [document.content for document in chunked_documents]
    )

    print(f"Generated {len(embeddings)} embeddings")

    print(f"Embedding dimension: {len(embeddings[0])}")

    # =====================================
    # VECTOR STORE
    # =====================================
    print("\nStoring in ChromaDB...")

    vector_store = ChromaStore()

    vector_store.add_documents(
        chunked_documents,
        embeddings,
    )

    print(f"Total vectors stored: " f"{vector_store.collection.count()}")

    # =====================================
    # BM25
    # =====================================
    print("\nBuilding BM25 index...")

    bm25 = BM25Retriever()

    bm25.index(chunked_documents)

    print("BM25 index ready")

    # =====================================
    # DENSE RETRIEVAL
    # =====================================
    print("\n====================")
    print("DENSE RETRIEVAL")
    print("====================")

    dense = DenseRetriever()

    dense_results = dense.retrieve(
        "AI engineering projects",
        top_k=3,
    )

    for i, document in enumerate(dense_results["documents"][0]):

        print(f"\nRESULT {i+1}")

        print(document[:250])

    # =====================================
    # BM25 RETRIEVAL
    # =====================================
    print("\n====================")
    print("BM25 RETRIEVAL")
    print("====================")

    sparse_results = bm25.retrieve(
        "AI engineering projects",
        top_k=3,
    )

    for i, (
        document,
        score,
    ) in enumerate(sparse_results):

        print(f"\nRESULT {i+1}")

        print(f"Score: {score}")

        print(document.content[:250])

    # =====================================
    # HYBRID RETRIEVAL
    # =====================================
    print("\n====================")
    print("HYBRID RETRIEVAL")
    print("====================")

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=bm25,
    )

    hybrid_results = hybrid.retrieve(
        "AI engineering projects",
        top_k=5,
    )

    for (
        chunk_id,
        score,
        text,
    ) in hybrid_results:

        print()

        print(f"Chunk ID: {chunk_id}")

        print(f"RRF Score: {score}")

        print(text[:250])


if __name__ == "__main__":
    main()
