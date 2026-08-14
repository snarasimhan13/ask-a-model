import chromadb

from src.embeddings import embed_texts, embed_query


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "research_papers"


def get_collection():
    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.get_or_create_collection(
        name="research_papers"
    )

    return collection

def collection_is_empty(collection) -> bool:
    return collection.count() == 0

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
        retrieved_chunks.append(
            {
                "text":
                    results["documents"][0][i],

                "source":
                    results["metadatas"][0][i][
                        "source"
                    ],

                "page":
                    results["metadatas"][0][i][
                        "page"
                    ],

                "distance":
                    results["distances"][0][i],
            }
        )

    return retrieved_chunks