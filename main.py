from src.loader import load_papers
from src.chunker import chunk_text
from src.embeddings import embed_texts
from src.retriever import retrieve
from src.generator import generate_answer


PAPER_DIRECTORY = "data/papers"


def main():

    print("\nLoading papers...\n")

    pages = load_papers(PAPER_DIRECTORY)

    print(f"\nExtracted {len(pages)} pages.")

    print("\nChunking papers...\n")

    chunks = chunk_text(
        pages,
        chunk_size=400,
        overlap=50,
    )

    print(f"Created {len(chunks)} chunks.")

    print("\nEmbedding chunks...")
    print("This can take a bit on the first run.\n")

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    chunk_embeddings = embed_texts(chunk_texts)

    print(
        f"Embedding matrix shape: "
        f"{chunk_embeddings.shape}"
    )

    print("\nAsk questions about your papers.")
    print("Type 'quit' to exit.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() in {
            "quit",
            "exit",
            "q",
        }:
            break

        if not question:
            continue

        retrieved_chunks = retrieve(
            question=question,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
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
                f"(similarity={chunk['score']:.3f})"
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