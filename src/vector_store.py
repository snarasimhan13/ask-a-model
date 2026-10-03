import chromadb

from src.embeddings import embed_texts, embed_query


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "research_papers"


def get_collection():
    """
    Return (client, collection). The client is needed later
    to delete/recreate the collection on --reset.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
    )

    return client, collection


def collection_is_empty(collection) -> bool:
    return collection.count() == 0


def reset_collection(client):
    """
    Delete the existing collection and return a fresh, empty one.

    Chroma keeps file handles open on the persistent directory,
    so we delete the collection through the client rather than
    removing chroma_db/ from disk.
    """
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # collection may not exist yet; that's fine
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
    )


def index_chunks(
    collection,
    chunks: list[dict],
):
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = embed_texts(texts)

    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk-{i}")

        metadatas.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "section": chunk["section"],
            }
        )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )


def retrieve(
    collection,
    question: str,
    top_k: int = 5,
) -> list[dict]:

    # obtain query embedding for question
    query_embedding = embed_query(question)

    # query embedding, gives closest to question embedding
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    # formats the info properly
    for i in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][i]
        retrieved_chunks.append(
            {
                "text": results["documents"][0][i],
                "source": metadata["source"],
                "page": metadata["page"],
                "section": metadata["section"],
                "distance": results["distances"][0][i],
            }
        )

    return retrieved_chunks
