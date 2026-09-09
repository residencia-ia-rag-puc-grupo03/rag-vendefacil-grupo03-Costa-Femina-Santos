from typing import Optional
from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    source_file: str
    doc_type: str
    chunk_id: str
    sensitivity: str


    customer_id: Optional[str] = None
    company_name: Optional[str] = None
    state: Optional[str] = None
    module: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    date: Optional[str] = None
    section: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}
