import ollama

GENERATION_MODEL = "gemma3:4b"


def build_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"""
        [SOURCE {i}]
        Paper: {chunk["source"]}
        Page: {chunk["page"]}
        Section = {chunk["section"]}

        {chunk["text"]}
        """
        )
    return "\n".join(context_parts)
    

def generate_answer(
        question: str,
        retrieved_chunks: list[dict],
) -> str:
    # get the formatted text context
    context = build_context(retrieved_chunks)

    system_prompt = """
    You are a research assistant answering questions about academic papers.

    You must answer ONLY using the provided research paper excerpts.

    Rules:

    1. Do not use outside knowledge.
    2. If the excerpts do not contain enough information to answer,
    say: "I don't have enough information in the retrieved papers."
    3. Every factual claim should include a citation.
    4. Cite sources using the paper filename and page number.
    5. Do not invent citations.

    Example citation:

    [attention.pdf, p. 4]
    """

    user_prompt = f"""
    QUESTION:

    {question}


    RESEARCH PAPER EXCERPTS:

    {context}


    Answer the question using only these excerpts.
    """

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response["message"]["content"]




