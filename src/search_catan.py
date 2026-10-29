from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

DB_PATH = "chroma_db"
COLLECTION_NAME = "rulebeacon_catan"

if not Path(DB_PATH).exists():
    raise FileNotFoundError(
        "Database not found. Run: python src/build_index.py"
    )

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)

question = input(
    "\nAsk RuleBeacon a CATAN rules question: "
).strip()

if not question:
    print("Please enter a question and try again.")
    raise SystemExit(1)


# Retrieve eight passages instead of four.
results = vector_store.similarity_search_with_score(
    question,
    k=8,
)

if not results:
    print("No passages were retrieved.")
    raise SystemExit(0)

print(f"\nRetrieved {len(results)} passages.")

# Look specifically for evidence about the value of a road.
found_exact_rule = False

for number, (document, distance) in enumerate(results, start=1):
    page_index = document.metadata.get("page")
    page_number = page_index + 1 if isinstance(page_index, int) else "unknown"
    text = document.page_content

    print(f"\n{'=' * 55}")
    print(f"RESULT {number} | PDF PAGE {page_number}")
    print(f"Distance: {distance:.4f}")
    print("=" * 55)
    print(text)

    if "roads are worth 0 vp" in text.lower():
        found_exact_rule = True

print("\n" + "=" * 55)
if found_exact_rule:
    print("CHECK PASSED: Found 'Roads are worth 0 VP.'")
else:
    print("CHECK: Exact road-scoring sentence not in these results.")
