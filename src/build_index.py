from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Load our API key from .env.
load_dotenv()

PDF_PATH = Path("data/rulebooks/catan_base_rules.pdf")
DB_PATH = "chroma_db"
COLLECTION_NAME = "rulebeacon_catan"


# Check that the rulebook exists.
if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"Rulebook not found: {PDF_PATH}"
    )


# Step 1: Read the PDF.
print("Reading CATAN rulebook...")

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()

print(f"Pages loaded: {len(pages)}")


# Step 2: Split the text into smaller passages.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
)

chunks = splitter.split_documents(pages)

# Keep useful information with every passage.
for chunk in chunks:
    chunk.metadata["game"] = "CATAN"
    chunk.metadata["edition"] = "Base game"

print(f"Chunks created: {len(chunks)}")


# Step 3: Set up the embedding model.
# LangChain uses document embeddings when indexing
# and query embeddings when searching by a question.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# Give each passage a stable ID so rerunning this
# script updates the same records instead of creating
# new IDs for every run.
chunk_ids = [
    f"catan-chunk-{i}"
    for i in range(len(chunks))
]


# Step 4: Store the passages and their embeddings.
print("Creating searchable database...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    ids=chunk_ids,
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
)

print("\nSUCCESS! Rulebook indexed.")
print("Chunks indexed:", len(chunks))
print("Records in database:", vector_store._collection.count())
print("Database location:", Path(DB_PATH).resolve())
