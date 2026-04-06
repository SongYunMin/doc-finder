from pydantic import BaseModel, Field


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1)


class TextSearchResponse(BaseModel):
    query: str
    answer: str
