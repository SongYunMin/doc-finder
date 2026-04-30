from __future__ import annotations

from pydantic import BaseModel, Field


class VisionTagRequest(BaseModel):
    filename: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    content_base64: str = Field(min_length=1)


class VisionTagResponse(BaseModel):
    keyword_tags: list[str] = Field(default_factory=list)
    normalized_tags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: str
