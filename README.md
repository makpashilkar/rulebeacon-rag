# RuleBeacon 🧭

**A retrieval-augmented generation (RAG) assistant for CATAN board-game rules.**

RuleBeacon lets players ask questions about the CATAN base game in plain English. It searches the official rulebook passages stored in a local Chroma vector database, then uses Google's Gemini model to generate an explanation with PDF page citations.

This is an educational portfolio project demonstrating document ingestion, text chunking, embeddings, semantic retrieval, grounded answer generation, source tracking, and a basic evaluation workflow.

> RuleBeacon is an independent, unofficial project. It is not affiliated with CATAN or its publisher. Consult the official rulebook when a ruling is unclear.

## Features

- Loads an English-language CATAN base-game rulebook from PDF.
- Splits the document into smaller passages while preserving page metadata.
- Generates text embeddings using Google's Gemini embedding model.
- Stores searchable passages locally using Chroma.
- Retrieves the eight most relevant passages for a question.
- Uses Gemini to generate an answer based on the retrieved evidence.
- Converts source IDs into PDF page citations using stored metadata.
- Provides a Streamlit chat interface with expandable retrieved passages.
- Includes basic retrieval checks for three example questions.

## How it works

```text
CATAN rulebook PDF
        |
        v
    PDF loading
        |
        v
 Text chunking + page metadata
        |
        v
  Gemini embeddings
        |
        v
 Local Chroma vector database
        |
 User asks a question
        |
        v
 Retrieve relevant passages
        |
        v
 Gemini generates an answer
        |
        v
 Answer with PDF page citations
```
