from abc import ABC, abstractmethod

from models.document import Document


class BaseLoader(ABC):

    @abstractmethod
    def load(
        self,
        file_path: str,
        original_filename: str,
    ) -> list[Document]:
        """
        Load a resource and convert it into
        a list of Document objects.
        """
        pass
