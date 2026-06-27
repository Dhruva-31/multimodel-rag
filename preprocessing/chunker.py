from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.document import Document
from models.metadata import Metadata


class Chunker:

    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: list[Document]) -> list[Document]:

        chunked_docs = []
        global_chunk_id = 0

        for document in documents:

            chunks = self.splitter.split_text(document.content)

            for chunk in chunks:

                metadata = Metadata(
                    source=document.metadata.source,
                    type=document.metadata.type,
                    page=document.metadata.page,
                    chunk_id=global_chunk_id,
                    chunk_size=len(chunk),
                )

                chunked_docs.append(Document(content=chunk, metadata=metadata))

                global_chunk_id += 1

        return chunked_docs
