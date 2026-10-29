from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load the API key from our .env file.
load_dotenv()

# Connect to Google's text-embedding model.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",
)

# Create a numerical fingerprint for a sample question.
question = "How do players build roads in CATAN?"
vector = embeddings.embed_query(question)

# Show the result without printing hundreds of numbers.
print("Embedding generation works!")
print("Number of dimensions:", len(vector))
print("First five numbers:", [round(x, 4) for x in vector[:5]])
