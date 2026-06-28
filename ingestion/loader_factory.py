from pathlib import Path

from ingestion.base_loader import BaseLoader
from ingestion.pdf_loader import PDFLoader
from ingestion.text_loader import TextLoader
from ingestion.docx_loader import DocxLoader
from ingestion.image_loader import ImageLoader


class LoaderFactory:

    _loaders: dict[str, type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".docx": DocxLoader,
        ".png": ImageLoader,
        ".jpg": ImageLoader,
        ".jpeg": ImageLoader,
        ".webp": ImageLoader,
    }

    @classmethod
    def get_loader(
        cls,
        file_path: str,
    ) -> BaseLoader:

        extension = Path(file_path).suffix.lower()

        loader = cls._loaders.get(extension)

        if loader is None:
            raise ValueError(f"Unsupported file type: {extension}")

        return loader()
