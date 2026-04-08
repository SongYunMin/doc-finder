from __future__ import annotations

from hashlib import sha256
from math import sqrt
import re


_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z]+|[가-힣]+")


class HashingEmbeddingService:
    def __init__(self, dimensions: int = 16) -> None:
        self._dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        if not tokens:
            return vector

        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            bucket = digest[0] % self._dimensions
            vector[bucket] += 1.0

        norm = sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector

        return [value / norm for value in vector]
