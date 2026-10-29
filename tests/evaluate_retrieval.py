from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

DB_PATH = "chroma_db"
COLLECTION_NAME = "rulebeacon_catan"


# Each test has a question and phrases that must appear
# somewhere in the retrieved passages.
TEST_CASES = [
    {
        "name": "Individual road victory points",
        "question": "How many victory points is a road worth?",
        "expected_terms": [
            "roads are worth 0 vp",
        ],
    },
    {
        "name": "Longest Route requirements",
        "question": "What are the requirements for the Longest Route tile?",
        "expected_terms": [
            "longest route",
            "5 continuous roads",
            "2 vps",
        ],
    },
    {
        "name": "Rolling a seven",
        "question": "What happens when a player rolls a 7?",
        "expected_terms": [
            "discard",
            "robber",
            "resource card",
        ],
    },
]


def main():
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

    passed = 0

    print("RULEBEACON RETRIEVAL EVALUATION")
    print("=" * 55)

    for test in TEST_CASES:
        results = vector_store.similarity_search(
            test["question"],
            k=8,
        )

        # Combine the retrieved text for checking.
        retrieved_text = "\n".join(
            document.page_content
            for document in results
        ).lower()

        missing_terms = [
            term
            for term in test["expected_terms"]
            if term.lower() not in retrieved_text
        ]

        print(f"\nTest: {test['name']}")
        print(f"Passages retrieved: {len(results)}")

        if missing_terms:
            print("RESULT: FAIL")
            print("Missing evidence terms:", ", ".join(missing_terms))

            pages = sorted({
                document.metadata["page"] + 1
                for document in results
                if isinstance(document.metadata.get("page"), int)
            })

            print("Retrieved PDF pages:", pages)
        else:
            print("RESULT: PASS")
            passed += 1

    print("\n" + "=" * 55)
    print(f"Score: {passed}/{len(TEST_CASES)} tests passed")

    if passed == len(TEST_CASES):
        print("All retrieval checks passed!")
    else:
        print("Some retrieval checks need investigation.")


if __name__ == "__main__":
    main()
