from src.loader import load_papers_interactive
from src.chunker import chunk_text
from src.vector_store import (
    get_collection,
    collection_is_empty,
    index_chunks,
    reset_collection,
    retrieve,
)
from src.generator import generate_answer


def build_index(collection) -> None:
    """
    Prompt the user for PDFs, chunk them, and add them
    to the given Chroma collection.
    """
    pages = load_papers_interactive()

    print(f"\nExtracted {len(pages)} pages.")

    chunks = chunk_text(
        pages,
        chunk_size=250,
    )

    print(f"Created {len(chunks)} chunks.")

    index_chunks(
        collection,
        chunks,
    )

    print("\nIndex saved to ChromaDB.")


def main():
    client, collection = get_collection()

    if collection_is_empty(collection):
        print("\nNo existing Chroma index found.")
        print("Let's add some papers.\n")
        build_index(collection)
    else:
        print(
            f"\nLoaded existing Chroma index "
            f"with {collection.count()} chunks."
        )

    print("\nAsk questions about your papers.")
    print("Type '--reset' to start over with new papers.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except EOFError:
            break

        if question.lower() in {"quit", "exit", "q"}:
            break

        if question == "--reset":
            print("\nClearing vector DB...")
            collection = reset_collection(client)
            print("Done. Let's add new papers.\n")
            build_index(collection)
            print(
                "\nAsk questions about your papers."
            )
            print(
                "Type '--reset' to start over, "
                "or 'quit' to exit.\n"
            )
            continue

        if not question:
            continue

        retrieved_chunks = retrieve(
            collection=collection,
            question=question,
            top_k=5,
        )

        print("\n--- Retrieved chunks ---\n")

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            print(
                f"{i}. "
                f"{chunk['source']} "
                f"page {chunk['page']} "
                f"| {chunk['section']} "
                f"| distance={chunk['distance']:.3f}"
            )

        print("\n--- Answer ---\n")

        answer = generate_answer(
            question,
            retrieved_chunks,
        )

        print(answer)

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
