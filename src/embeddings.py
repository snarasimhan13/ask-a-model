import numpy as np
import ollama

EMBEDDING_MODEL = "embeddinggemma"

# embeddings for the pieces of text
def embed_texts(texts: list[str]) -> np.ndarray:
    all_embeddings = []

    batch_size = 10

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        print(
            f"Embedding chunks "
            f"{start + 1}-{min(start + batch_size, len(texts))} "
            f"of {len(texts)}"
        )

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        all_embeddings.extend(response["embeddings"])

    return np.array(
        all_embeddings,
        dtype=np.float32,
    )

# create embedding for the query
def embed_query(query: str):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query,
    )

    return np.array(
        response["embeddings"][0],
        dtype=np.float32,
    )