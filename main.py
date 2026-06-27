from ingestion.pdf_loader import PDFLoader
from preprocessing.chunker import Chunker

from embeddings.dense_embeddings import DenseEmbedder

from vectorstore.chroma_store import ChromaStore

from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

from sparse.bm25_index import BM25Retriever

QUERY = "What AI engineering projects help get jobs?"


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

    embeddings = embedder.embed_documents([doc.content for doc in chunked_documents])

    print(f"Generated {len(embeddings)} embeddings")

    print(f"Embedding dimension: " f"{len(embeddings[0])}")

    # =====================================
    # CHROMA
    # =====================================
    print("\nBuilding ChromaDB...")

    vector_store = ChromaStore()

    vector_store.add_documents(
        chunked_documents,
        embeddings,
    )

    print(f"Stored " f"{vector_store.collection.count()}" f" vectors")

    # =====================================
    # BM25
    # =====================================
    print("\nBuilding BM25...")

    bm25 = BM25Retriever()

    bm25.index(chunked_documents)

    print("BM25 ready")

    # =====================================
    # DENSE
    # =====================================
    dense = DenseRetriever()

    # =====================================
    # HYBRID
    # =====================================
    print("\n========================")
    print("HYBRID RETRIEVAL")
    print("========================")

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=bm25,
    )

    hybrid_results = hybrid.retrieve(
        QUERY,
        top_k=10,
    )

    for (
        chunk_id,
        score,
        text,
    ) in hybrid_results:

        print()
        print(f"Chunk: {chunk_id}")

        print(f"RRF: {score:.5f}")

        print(text[:200])

    # =====================================
    # RERANKER
    # =====================================
    print("\n========================")
    print("CROSS ENCODER")
    print("========================")

    reranker = CrossEncoderReranker()

    documents = [text for _, _, text in hybrid_results]

    reranked = reranker.rerank(
        QUERY,
        documents,
        top_k=5,
    )

    for rank, (
        document,
        score,
    ) in enumerate(
        reranked,
        start=1,
    ):

        print()
        print(f"Rank: {rank}")

        print(f"Score: {score:.4f}")

        print(document[:250])


if __name__ == "__main__":
    main()
