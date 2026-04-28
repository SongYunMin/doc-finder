from pydantic import BaseModel, Field


class TagSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    review_status: str | None = None


class CmsReference(BaseModel):
    unit_id: int
    data_id: int


class TagSearchResult(BaseModel):
    unit_id: int
    data_id: int
    image_id: int
    asset_path: str
    preview_path: str | None = None
    matched_tags: list[str] = Field(default_factory=list)
    matched_display_tags: list[str] = Field(default_factory=list)
    confidence: float
    score: float
    cms_ref: CmsReference


class TagSearchResponse(BaseModel):
    results: list[TagSearchResult] = Field(default_factory=list)
