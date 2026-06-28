from ingestion.loader_factory import LoaderFactory

from preprocessing.chunker import Chunker

from embeddings.dense_embeddings import DenseEmbedder

from vectorstore.chroma_store import ChromaStore

from sparse.bm25_index import BM25Retriever


def build_knowledge_base(
    file_path: str,
    original_filename: str,
):

    loader = LoaderFactory.get_loader(file_path)

    documents = loader.load(file_path, original_filename)

    if not documents:
        raise ValueError("No text could be extracted from this file.")

    chunker = Chunker()

    chunked_documents = chunker.chunk(documents)

    if not chunked_documents:
        raise ValueError("No searchable content found in the uploaded file.")

    embedder = DenseEmbedder()

    embeddings = embedder.embed_documents([doc.content for doc in chunked_documents])

    vector_store = ChromaStore()

    vector_store.add_documents(
        chunked_documents,
        embeddings,
    )

    bm25_retriever = BM25Retriever()

    bm25_retriever.index(chunked_documents)

    return bm25_retriever, vector_store
