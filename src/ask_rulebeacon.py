import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma


# Load the API key from .env.
load_dotenv()

DB_PATH = "chroma_db"
COLLECTION_NAME = "rulebeacon_catan"


# Check that the database exists.
if not Path(DB_PATH).exists():
    raise FileNotFoundError(
        "Database not found. Run: python src/build_index.py"
    )


# Connect to the embedding model used to build the index.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Open the existing Chroma database.
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
)

# Connect to the working Gemini chat model.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
)


def extract_response_text(response):
    """Extract readable text without printing provider metadata."""
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text_parts.append(block["text"])

        return "\n".join(text_parts).strip()

    return str(content)


def replace_source_ids(answer, source_pages):
    """Convert model-provided source IDs into verified PDF pages."""

    def replace_match(match):
        source_id = match.group(1)
        page_number = source_pages.get(source_id)

        if page_number is None:
            return f"[unverified source: {source_id}]"

        return f"[PDF p. {page_number}]"

    return re.sub(r"\[(S\d+)\]", replace_match, answer)


# Ask the player a question.
question = input(
    "\nWelcome to RuleBeacon! Ask a CATAN rules question: "
).strip()

if not question:
    print("Please enter a question.")
    raise SystemExit(1)


# Step 1: Retrieve relevant passages.
results = vector_store.similarity_search_with_score(
    question,
    k=15,
)

if not results:
    print("No passages were retrieved.")
    raise SystemExit(0)


# Step 2: Label every passage and record its real PDF page.
context_parts = []
source_pages = {}

for number, (document, distance) in enumerate(results, start=1):
    source_id = f"S{number}"
    page_index = document.metadata.get("page")

    if isinstance(page_index, int):
        page_number = page_index + 1
        page_label = f"PDF page {page_number}"
        source_pages[source_id] = page_number
    else:
        page_label = "page unknown"
        source_pages[source_id] = None

    context_parts.append(
        f"[{source_id}] CATAN Base Game Rulebook, {page_label}\n"
        f"{document.page_content}"
    )

context = "\n\n---\n\n".join(context_parts)


# Step 3: Ask Gemini to answer using the retrieved evidence.
prompt = f"""
You are RuleBeacon, a friendly CATAN rules assistant.

Answer the player's question using ONLY the source passages below.

Instructions:
1. Explain the answer in simple English.
2. Cite each important rule claim using the source IDs given
   in the passages, exactly like [S1] or [S2].
3. If multiple sources support a claim, cite them separately,
   for example [S1] [S2].
4. Do NOT write PDF page numbers yourself. The program will
   convert source IDs into page citations.
5. Do not invent rules or use outside knowledge to fill gaps.
6. If the passages do not support an answer, say that you
   could not verify it from the retrieved passages.
7. If relevant information is missing or unclear, explain that.
8. Keep the answer concise.

PLAYER QUESTION:
{question}

RETRIEVED SOURCES:
{context}

ANSWER:
"""


# Step 4: Generate the answer.
print("\nRuleBeacon is checking the rulebook...\n")

response = llm.invoke(prompt)
answer = extract_response_text(response)

# Step 5: Convert source IDs into actual PDF page references.
answer = replace_source_ids(answer, source_pages)
# Collapse repeated citations to the same page when adjacent.
answer = re.sub(
    r"(\[PDF p\. \d+\])(?:\s+\1)+",
    r"\1",
    answer,
)
print("=" * 55)
print("RULEBEACON'S ANSWER")
print("=" * 55)
print(answer or "Gemini returned no readable text.")

print("\nRetrieved source pages:")
for source_id, page_number in source_pages.items():
    page_label = (
        f"PDF page {page_number}"
        if page_number is not None
        else "page unknown"
    )
    print(f"- {source_id}: {page_label}")
