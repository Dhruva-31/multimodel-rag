from models.document import Document
from models.metadata import Metadata
from ingestion.base_loader import BaseLoader


class TextLoader(BaseLoader):

    def load(
        self,
        file_path: str,
        original_filename: str,
    ) -> list[Document]:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:

            text = f.read()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        documents = []

        for i, paragraph in enumerate(paragraphs):

            documents.append(
                Document(
                    content=paragraph,
                    metadata=Metadata(
                        source=file_path,
                        filename=original_filename,
                        type="txt",
                        paragraph=i + 1,
                    ),
                )
            )

        return documents
