from pydantic import BaseModel

class SingleQAResponse(BaseModel):
    answer: str
