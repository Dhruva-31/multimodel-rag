from dataclasses import dataclass
from models.metadata import Metadata


@dataclass
class Document:
    content: str
    metadata: Metadata
