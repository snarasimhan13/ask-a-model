from src.loader import load_papers
from src.chunker import chunk_text
from src.vector_store import (
    get_collection,
    collection_is_empty,
    index_chunks,
    retrieve,
)
from src.generator import generate_answer

PAPER_DIRECTORY = "data/papers"


def main():

    collection = get_collection()

    if collection_is_empty(collection):

        print("\nNo existing Chroma index found.")
        print("Building index...\n")

        pages = load_papers(
            PAPER_DIRECTORY
        )

        print(
            f"\nExtracted {len(pages)} pages."
        )

        chunks = chunk_text(
            pages,
            chunk_size=400,
            overlap=50,
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        index_chunks(
            collection,
            chunks,
        )

        print(
            "\nIndex saved to ChromaDB."
        )

    else:

        print(
            f"\nLoaded existing Chroma index "
            f"with {collection.count()} chunks."
        )

    print("\nAsk questions about your papers.")
    print("Type 'quit' to exit.\n")

    while True:

        question = input(
            "Question: "
        ).strip()

        if question.lower() in {
            "quit",
            "exit",
            "q",
        }:
            break

        if not question:
            continue

        retrieved_chunks = retrieve(
            collection=collection,
            question=question,
            top_k=5,
        )

        print(
            "\n--- Retrieved chunks ---\n"
        )

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):

            print(
                f"{i}. "
                f"{chunk['source']} "
                f"page {chunk['page']} "
                f"(distance="
                f"{chunk['distance']:.3f})"
            )

        print("\n--- Answer ---\n")

        answer = generate_answer(
            question,
            retrieved_chunks,
        )

        print(answer)

        print(
            "\n" + "=" * 80 + "\n"
        )


if __name__ == "__main__":
    main()