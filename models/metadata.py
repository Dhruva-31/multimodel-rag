from dataclasses import dataclass


@dataclass
class Metadata:
    source: str
    filename: str
    type: str

    page: int | None = None
    paragraph: int | None = None

    chunk_id: int | None = None
    chunk_size: int | None = None
