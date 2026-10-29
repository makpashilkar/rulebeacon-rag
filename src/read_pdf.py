from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("data/rulebooks/catan_base_rules.pdf")

if not pdf_path.exists():
    print("PDF not found. Check the filename inside the data folder.")
    raise SystemExit(1)

reader = PdfReader(str(pdf_path))

print("Number of pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages[:2], start=1):
    text = page.extract_text() or ""

    print(f"\n--- Page {page_number} ---")
    print(text[:2000] if text.strip() else "No extractable text found.")