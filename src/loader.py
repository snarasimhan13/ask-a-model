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

# this is for each paper, calls load_pdf() for each pdf
def load_papers(directory: str) -> list[dict]:
    directory = Path(directory)

    total_pages = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        print(f"Loading {pdf_path.name}...")

        pages = load_pdf(pdf_path)
        total_pages.extend(pages)
    return total_pages

