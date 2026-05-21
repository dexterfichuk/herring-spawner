import numpy as np

from herring_spawner.embeddings.search import search_similar


def test_search_similar_returns_cosine_ranked_neighbors():
    query = np.array([1.0, 0.0])
    embeddings = {
        "spawn-a": np.array([0.9, 0.1]),
        "water-b": np.array([0.0, 1.0]),
        "spawn-c": np.array([0.8, 0.2]),
    }
    results = search_similar(query=query, embeddings=embeddings, limit=2)
    assert [result.item_id for result in results] == ["spawn-a", "spawn-c"]
    assert results[0].score > results[1].score
