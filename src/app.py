import re
from pathlib import Path

import streamlit as st
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


# Configure the website.
st.set_page_config(
    page_title="RuleBeacon | CATAN Rules Assistant",
    page_icon="🧭",
    layout="centered",
)


# Reuse model connections during Streamlit reruns.
@st.cache_resource
def load_components():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(
            "Rulebook database not found. "
            "Run: python src/build_index.py"
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )

    return vector_store, llm


def extract_response_text(response):
    """Extract readable text from Gemini's response."""

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                text = block.get("text", "")

                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts).strip()

    return str(content)


def replace_source_ids(answer, source_pages):
    """
    Convert source IDs into verified PDF page citations.

    Supports both [S1] and grouped citations such as [S1, S4].
    """

    # Match [S1], [S2, S4], or [S1, S3, S8].
    grouped_citation_pattern = (
        r"\[((?:S\d+)(?:\s*,\s*S\d+)*)\]"
    )

    def replace_group(match):
        source_ids = re.findall(
            r"S\d+",
            match.group(1),
        )

        page_citations = []

        for source_id in source_ids:
            page_number = source_pages.get(source_id)

            if page_number is None:
                page_citations.append(
                    f"[unverified source: {source_id}]"
                )
            else:
                page_citations.append(
                    f"[PDF p. {page_number}]"
                )

        # Remove duplicate page numbers within this citation group.
        unique_citations = list(dict.fromkeys(page_citations))

        return " ".join(unique_citations)

    answer = re.sub(
        grouped_citation_pattern,
        replace_group,
        answer,
    )

    # Remove repeated adjacent citations to the same page.
    answer = re.sub(
        r"(\[PDF p\. \d+\])(?:\s+\1)+",
        r"\1",
        answer,
    )

    return answer


def answer_question(question, vector_store, llm):
    """Retrieve evidence and generate a grounded answer."""

    results = vector_store.similarity_search_with_score(
        question,
        k=8,
    )

    if not results:
        return (
            "I couldn't verify that from the retrieved "
            "rulebook passages.",
            [],
        )

    context_parts = []
    source_pages = {}
    displayed_sources = []

    for number, (document, distance) in enumerate(
        results,
        start=1,
    ):
        source_id = f"S{number}"
        page_index = document.metadata.get("page")

        if isinstance(page_index, int):
            # PDF loader metadata starts counting pages at zero.
            page_number = page_index + 1
            page_label = f"PDF page {page_number}"
            source_pages[source_id] = page_number
        else:
            page_number = None
            page_label = "page unknown"
            source_pages[source_id] = None

        context_parts.append(
            f"[{source_id}] CATAN Base Game Rulebook, "
            f"{page_label}\n"
            f"{document.page_content}"
        )

        displayed_sources.append(
            {
                "source_id": source_id,
                "page": page_number,
                "distance": distance,
                "text": document.page_content,
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are RuleBeacon, a friendly CATAN rules assistant.

Answer the player's question using ONLY the passages below.

Instructions:
1. Explain the answer in simple English.
2. Cite supporting claims with source IDs.
3. You may cite one source as [S1] or multiple sources as [S1, S4].
4. Cite only passages that directly support the claim.
5. Never write PDF page numbers yourself.
6. Do not use outside knowledge to fill gaps in the evidence.
7. If the passages do not support an answer, say:
   "I couldn't verify that from the retrieved rulebook passages."
8. Explain when the evidence is incomplete or unclear.
9. Keep the answer concise.

PLAYER QUESTION:
{question}

RETRIEVED RULEBOOK PASSAGES:
{context}

ANSWER:
"""

    response = llm.invoke(prompt)
    answer = extract_response_text(response)

    # Python, not Gemini, determines the cited page numbers.
    answer = replace_source_ids(answer, source_pages)

    return answer, displayed_sources


# Initialise the chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []


# App heading.
st.title("🧭 RuleBeacon")
st.subheader("Your CATAN rules referee")

st.write(
    "Ask a rules question in plain English. "
    "RuleBeacon searches the base-game rulebook, "
    "explains the relevant rule, and shows its sources."
)

st.caption(
    "Unofficial prototype · CATAN base game only · "
    "Check the official rulebook if a ruling is unclear."
)

st.divider()


# Display previous messages and their retrieved sources.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            with st.expander(
                "Inspect retrieved rulebook passages"
            ):
                st.caption(
                    "Retrieved passages are search results. "
                    "Not every passage necessarily supports "
                    "the final answer."
                )

                for source in message["sources"]:
                    if source["page"] is not None:
                        page_label = (
                            f"PDF page {source['page']}"
                        )
                    else:
                        page_label = "Page unknown"

                    st.markdown(
                        f"**{source['source_id']} · "
                        f"{page_label}** · "
                        f"Distance: {source['distance']:.4f}"
                    )

                    st.text(source["text"][:1500])
                    st.divider()


# Chat input.
question = st.chat_input(
    "Ask a CATAN rules question..."
)

if question and question.strip():
    question = question.strip()

    # Display the user's question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve the rulebook passages and answer.
    with st.chat_message("assistant"):
        try:
            vector_store, llm = load_components()

            with st.spinner("Searching the rulebook..."):
                answer, sources = answer_question(
                    question,
                    vector_store,
                    llm,
                )

            st.markdown(answer)

            if sources:
                with st.expander(
                    "Inspect retrieved rulebook passages"
                ):
                    st.caption(
                        "Review the source text to verify "
                        "whether it supports each claim."
                    )

                    for source in sources:
                        if source["page"] is not None:
                            page_label = (
                                f"PDF page {source['page']}"
                            )
                        else:
                            page_label = "Page unknown"

                        st.markdown(
                            f"**{source['source_id']} · "
                            f"{page_label}** · "
                            f"Distance: {source['distance']:.4f}"
                        )

                        st.text(source["text"][:1500])
                        st.divider()

        except Exception as error:
            answer = (
                "I couldn't complete that request. "
                "Check the error details and try again."
            )
            sources = []

            st.error(f"Request failed: {error}")
            st.markdown(answer)

    # Save the response in the conversation.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
