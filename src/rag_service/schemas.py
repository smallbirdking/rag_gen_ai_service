from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = "default"
    history: List[Message] = Field(default_factory=list)
    retrieval_mode: Optional[Literal["vector_only", "hybrid"]] = None
    reranker_enabled: Optional[bool] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    score: float


class AskResponse(BaseModel):
    request_id: str
    answer: str
    citations: List[Citation]
    refusal: bool
    refusal_reason: Optional[str] = None
    confidence: float
    latency_ms: int
    token_usage: Dict[str, int]
    cache_hit: bool
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
