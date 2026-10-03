from pathlib import Path
from pypdf import PdfReader

# pdf loader


def load_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages = []

    # stores text from each page in the pdf
    # stores in pages, dict. list: text, page, source
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
        pages.append(
            {
                "text": text,
                "page": page_no,
                "source": pdf_path.name,
            }
        )
    return pages


def load_papers(directory: str) -> list[dict]:
    directory = Path(directory)

    total_pages = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        print(f"Loading {pdf_path.name}...")

        pages = load_pdf(pdf_path)
        total_pages.extend(pages)
    return total_pages


def load_papers_interactive() -> list[dict]:
    """
    Prompt the user for PDF paths one by one and load each.

    Enter a blank line to finish. Requires at least one paper.
    """

    print(
        "Enter PDF paths one per line. "
        "Press Enter on a blank line when done."
    )

    collected_pages: list[dict] = []
    seen_paths: set[Path] = set()
    paper_count = 0

    while True:
        try:
            raw = input("PDF path: ").strip()
        except EOFError:
            raw = ""

        # strip surrounding quotes that Finder drag-and-drop
        # sometimes adds on paths containing spaces
        raw = raw.strip("'").strip('"').strip()

        if not raw:
            if paper_count == 0:
                print("Please add at least one paper.")
                continue
            break

        path = Path(raw).expanduser()

        if not path.exists():
            print(f"  ! Path not found: {path}")
            continue

        if not path.is_file():
            print(f"  ! Not a file: {path}")
            continue

        if path.suffix.lower() != ".pdf":
            print(f"  ! Not a PDF: {path}")
            continue

        resolved = path.resolve()
        if resolved in seen_paths:
            print(f"  ! Already added: {path.name}")
            continue

        pages = load_pdf(path)

        if not pages:
            print(
                f"  ! No extractable text in {path.name} "
                "(scanned or image-only?). Skipped."
            )
            continue

        collected_pages.extend(pages)
        seen_paths.add(resolved)
        paper_count += 1

        print(f"  + Added {path.name} ({len(pages)} pages).")

    return collected_pages
