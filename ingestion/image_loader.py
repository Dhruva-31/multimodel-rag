from PIL import Image
import pytesseract
from models.document import Document
from models.metadata import Metadata
from ingestion.base_loader import BaseLoader


class ImageLoader(BaseLoader):

    def load(
        self,
        file_path: str,
        original_filename: str,
    ) -> list[Document]:

        image = Image.open(file_path)

        text = pytesseract.image_to_string(image)

        return [
            Document(
                content=text,
                metadata=Metadata(
                    source=file_path,
                    filename=original_filename,
                    type="image",
                ),
            )
        ]
