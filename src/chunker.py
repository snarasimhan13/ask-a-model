import re


# detects if new chunk should be started for new section
def is_section_heading(text: str) -> bool:

    text = text.strip()

    if not text:
        return False


    numbered_heading = re.match(
        r"^\d+(\.\d+)*\.?\s+[A-Z].*",
        text
    )

    if numbered_heading:
        return True


    common_headings = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "methods",
        "method",
        "methodology",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
    }

    if text.lower() in common_headings:
        return True

    # Short ALL-CAPS lines may also be headings
    if (
        len(text.split()) <= 8
        and text.isupper()
        and len(text) > 2
    ):
        return True

    return False

def split_into_sections(text: str) -> list[dict]:
    lines = text.splitlines()
    sections = []

    curr_section = ""
    curr_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            curr_lines.append("")
            continue

        # if starting new heading
        if is_section_heading(line):
            if curr_lines:
                section_text = "\n".join(curr_lines).strip()
                if section_text:
                    sections.append(
                        {
                            "section": curr_section,
                            "text": section_text,
                        }
                    )
            curr_section = line
            curr_lines = []
        else:
            curr_lines.append(line)
    if curr_lines:
        section_text = "\n".join(
            curr_lines
        ).strip()

        if section_text:
            sections.append(
                {
                    "section": curr_section,
                    "text": section_text,
                }
            )

    return sections

def split_into_paragraphs(text: str) -> list[str]:
    # if there is a \n
    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    cleaned = []

    for paragraph in paragraphs:
        paragraph = " ".join(
            paragraph.split()
        ).strip()

        if paragraph:
            cleaned.append(paragraph)

    return cleaned

def chunk_paragraphs(
    paragraphs: list[str],
    max_words: int = 250,
) -> list[str]:
    """
    Combine paragraphs until max_words is reached.

    Paragraphs are kept intact whenever possible.
    """

    chunks = []

    current_paragraphs = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        paragraph_length = len(paragraph_words)

        # If adding this paragraph would make
        # the chunk too large, save current chunk.
        if (
            current_paragraphs
            and
            current_word_count + paragraph_length
            > max_words
        ):
            chunks.append(
                "\n\n".join(
                    current_paragraphs
                )
            )

            current_paragraphs = []
            current_word_count = 0

        # If one paragraph itself is huge,
        # fall back to splitting it by words.
        if paragraph_length > max_words:
            words = paragraph_words

            for start in range(
                0,
                len(words),
                max_words
            ):
                piece = words[
                    start:start + max_words
                ]

                if current_paragraphs:
                    chunks.append(
                        "\n\n".join(
                            current_paragraphs
                        )
                    )

                    current_paragraphs = []
                    current_word_count = 0

                chunks.append(
                    " ".join(piece)
                )

        else:
            current_paragraphs.append(
                paragraph
            )

            current_word_count += (
                paragraph_length
            )

    # Save anything remaining
    if current_paragraphs:
        chunks.append(
            "\n\n".join(
                current_paragraphs
            )
        )

    return chunks


def chunk_text(
    pages: list[dict],
    chunk_size: int = 250,
) -> list[dict]:
    """
    Convert PDF pages into section-aware,
    paragraph-aware chunks.
    """

    chunks = []

    chunk_id = 0

    for page in pages:

        sections = split_into_sections(
            page["text"]
        )

        for section in sections:

            paragraphs = split_into_paragraphs(
                section["text"]
            )

            section_chunks = chunk_paragraphs(
                paragraphs,
                max_words=chunk_size,
            )

            for text in section_chunks:

                chunks.append(
                    {
                        "id": chunk_id,
                        "text": text,
                        "source": page["source"],
                        "page": page["page"],
                        "section": section["section"],
                    }
                )

                chunk_id += 1

    return chunks