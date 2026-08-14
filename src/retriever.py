import numpy as np

from src.embeddings import embed_query


def cosine_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between one query vector
    and every document/chunk vector.
    """


    query_norm = np.linalg.norm(query_embedding)

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1,
    )

    # basically finding the best matches between each of the
    # document embedding and the query embedding
    similarities = (
        document_embeddings @ query_embedding
    ) / (document_norms * query_norm + 1e-10)

    return similarities


def retrieve(
    question: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    top_k: int = 5,
) -> list[dict]:

    query_embedding = embed_query(question)

    scores = cosine_similarity(
        query_embedding,
        chunk_embeddings,
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])

        results.append(result)

    return results