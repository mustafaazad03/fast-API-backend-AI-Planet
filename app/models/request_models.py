from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    pdf_id: str
    question: str

class HistoryResponse(BaseModel):
    history: List[dict]
