from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Find our CATAN rulebook.
pdf_path = Path("data/rulebooks/catan_base_rules.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(
        f"Could not find the rulebook at: {pdf_path}"
    )


# Load the PDF, keeping track of its pages.
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

print("Pages loaded:", len(pages))


# Break the text into smaller passages.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
)

chunks = splitter.split_documents(pages)

print("Chunks created:", len(chunks))


# Display the first five chunks so we can inspect them.
for number, chunk in enumerate(chunks[:5], start=1):
    page_number = chunk.metadata.get("page")

    # PDF page numbers in metadata start at zero.
    if page_number is not None:
        page_number += 1

    print(f"\n{'=' * 50}")
    print(f"CHUNK {number} | PDF PAGE {page_number}")
    print("=" * 50)
    print(chunk.page_content[:650])
