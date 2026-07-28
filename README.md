# Equity Research Report Intelligence (Multimodal RAG)

## Overview

A production-style Multimodal Retrieval-Augmented Generation (RAG)
system for analyzing equity research reports. The application extracts
and indexes **text**, **tables**, and **charts** from PDF reports,
stores them in **ChromaDB**, and answers questions using **Groq Llama
3.3** with support for **multi-document retrieval** and **cross-encoder
reranking**.

## Features

-   Text extraction and semantic chunking
-   Table extraction using pdfplumber
-   Chart understanding using Gemini Vision
-   ChromaDB vector database
-   Multi-document search across reports
-   Duplicate upload prevention
-   Delete individual reports from the vector database
-   Report-level filtering
-   Cross-document comparison prompting
-   Two-stage retrieval:
    -   Bi-Encoder (Sentence Transformers)
    -   Cross-Encoder reranking (MS MARCO MiniLM)

## Architecture

``` text
PDF Reports
     │
     ▼
Text ─ Tables ─ Charts
     │
     ▼
Embedding (all-MiniLM-L6-v2)
     │
     ▼
ChromaDB
     │
     ▼
Top-50 Candidate Retrieval
     │
     ▼
Cross-Encoder Reranking
     │
     ▼
Top-5 Relevant Chunks
     │
     ▼
Groq Llama 3.3
     │
     ▼
Final Answer with Source Citations
```

## Tech Stack

-   Python
-   Streamlit
-   ChromaDB
-   Sentence Transformers
-   CrossEncoder (MS MARCO MiniLM)
-   LangChain Text Splitters
-   PyMuPDF
-   pdfplumber
-   Google Gemini Vision
-   Groq API

## Project Layers

### Layer 1 - Text RAG

-   PDF text extraction
-   Recursive chunking
-   Embeddings
-   Vector storage

### Layer 2 - Table Intelligence

-   Table extraction
-   Markdown conversion
-   Table indexing

### Layer 3 - Chart Intelligence

-   Exhibit detection
-   Chart cropping
-   Gemini Vision descriptions
-   Image indexing

### Layer 4 - Multi-Document Intelligence

-   Duplicate prevention
-   Report management
-   Delete reports
-   Report filtering
-   Cross-document prompting

### Layer 5 - Retrieval Optimization

-   Retrieve top 50 candidates
-   Cross-encoder reranking
-   Return top 5 most relevant chunks

## Folder Structure

``` text
equity-research-rag/
├── app.py
├── ingestion.py
├── retrieval.py
├── generation.py
├── config.py
├── requirements.txt
├── chroma_db/
├── sample_reports/
└── README.md
```
## Example Questions

-   Compare the target price and analyst rating for Infosys and TCS.
-   What is the revenue outlook for Infosys?
-   Summarize key risks mentioned in the TCS report.
-   Compare margin guidance across both reports.

## License

This project is intended for learning, research, and portfolio purposes.
