from ingestion.pdf_loader import PDFLoader
from preprocessing.chunker import Chunker
from embeddings.dense_embeddings import DenseEmbedder
from vectorstore.chroma_store import ChromaStore
from retrieval.dense_retriever import DenseRetriever
from sparse.bm25_index import BM25Retriever


def main():

    # =====================================
    # Ingestion
    # =====================================
    print("Loading PDF...")
    loader = PDFLoader()

    documents = loader.load("uploads/sample.pdf")

    print(f"Loaded {len(documents)} pages")

    # =====================================
    # Chunking
    # =====================================
    print("\nChunking...")
    chunker = Chunker()

    chunked_documents = chunker.chunk(documents)

    print(f"Created {len(chunked_documents)} chunks")

    # =====================================
    # Embeddings
    # =====================================
    print("\nGenerating embeddings...")
    embedder = DenseEmbedder()

    embeddings = embedder.embed_documents([doc.content for doc in chunked_documents])

    print(f"Generated {len(embeddings)} embeddings")

    print(f"Embedding dimension: {len(embeddings[0])}")

    # =====================================
    # Chroma
    # =====================================
    print("\nStoring in ChromaDB...")

    vector_store = ChromaStore()

    vector_store.add_documents(chunked_documents, embeddings)

    print(f"Total vectors stored: " f"{vector_store.collection.count()}")

    # =====================================
    # Dense Retrieval
    # =====================================
    print("\n==============================")
    print("DENSE RETRIEVAL")
    print("==============================")

    dense_retriever = DenseRetriever()

    dense_results = dense_retriever.retrieve("AI engineering projects", top_k=3)

    for i, doc in enumerate(dense_results["documents"][0]):
        print(f"\nRESULT {i+1}")
        print(doc[:300])

    # =====================================
    # BM25 Retrieval
    # =====================================
    print("\n==============================")
    print("BM25 RETRIEVAL")
    print("==============================")

    bm25 = BM25Retriever()

    bm25.index(chunked_documents)

    sparse_results = bm25.retrieve("AI engineering projects", top_k=3)

    for i, (doc, score) in enumerate(sparse_results):
        print(f"\nRESULT {i+1}")
        print(f"Score: {score}")
        print(doc.content[:300])

    # =====================================
    # Sample chunk
    # =====================================
    print("\n==============================")
    print("SAMPLE CHUNK")
    print("==============================")

    print(chunked_documents[0].metadata)

    print(chunked_documents[0].content[:300])


if __name__ == "__main__":
    main()
