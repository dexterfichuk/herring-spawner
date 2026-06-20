from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    item_id: str
    score: float


def search_similar(
    query: np.ndarray, embeddings: dict[str, np.ndarray], limit: int = 10
) -> list[SearchResult]:
    query_norm = _normalize(query)
    results = [
        SearchResult(
            item_id=item_id, score=float(np.dot(query_norm, _normalize(vector)))
        )
        for item_id, vector in embeddings.items()
    ]
    return sorted(results, key=lambda r: r.score, reverse=True)[:limit]


def _normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(values)
    if norm == 0:
        return values
    return values / norm
