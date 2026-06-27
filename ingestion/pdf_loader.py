import fitz

from models.document import Document
from models.metadata import Metadata


class PDFLoader:

    def load(self, file_path: str) -> list[Document]:

        pdf: fitz.Document = fitz.open(file_path)
        documents = []

        for page_num in range(len(pdf)):
            page = pdf[page_num]

            text = page.get_text("text")

            documents.append(
                Document(
                    content=text,
                    metadata=Metadata(
                        source=file_path,
                        type="pdf",
                        page=page_num + 1,
                    ),
                )
            )

        pdf.close()

        return documents
