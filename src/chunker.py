
# chunking text by words, with a 50 word overlap
def chunk_text(pages: list[dict], chunk_size: int = 400, overlap: int = 50,) -> list[dict]:
   
    chunks = []

    for page in pages:
        words = page["text"].split()

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunks.append(
                {
                    "text": " ".join(chunk_words),
                    "source": page["source"],
                    "page": page["page"],
                }
            )

            start += chunk_size - overlap

    return chunks