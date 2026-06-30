# 🧠 Multimodal RAG System

> A production-inspired Retrieval-Augmented Generation (RAG) system supporting PDFs, documents, images, and text files using hybrid retrieval, reranking, source grounding, and hallucination guardrails.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-green)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-orange)
![BM25](https://img.shields.io/badge/BM25-Sparse%20Retrieval-purple)
![RAG](https://img.shields.io/badge/RAG-Hybrid-black)

---

## 🚀 Overview

This project implements a **multimodal Retrieval-Augmented Generation (RAG) pipeline** capable of ingesting and querying multiple document formats while minimizing hallucinations through retrieval grounding and citation validation.

The system combines:

* **Dense semantic retrieval**
* **Sparse keyword retrieval (BM25)**
* **Reciprocal Rank Fusion (RRF)**
* **Cross-encoder reranking**
* **Input and output guardrails**
* **Source citation validation**

The goal is to improve retrieval accuracy, response quality, and factual grounding compared to traditional vector-only RAG systems.

---

## ✨ Features

### 📂 Multi-format Document Ingestion

Supports:

* PDF files
* Text files
* DOCX documents
* Images (OCR extraction)

---

### 🔍 Hybrid Retrieval

The retrieval pipeline combines:

* Dense semantic search using embeddings
* Sparse keyword search using BM25
* Reciprocal Rank Fusion (RRF) ranking

This improves both semantic understanding and exact keyword matching.

---

### 🎯 Cross-Encoder Reranking

Retrieved candidates are reranked using a cross-encoder model to improve relevance before generation.

---

### 🛡️ Guardrails

The system includes:

* **Input guardrails**

  * Prompt injection detection
  * Unsafe query filtering

* **Output guardrails**

  * Citation verification
  * Hallucination detection
  * Source grounding checks

---

### 📚 Source Attribution

Generated responses include validated citations linked to the retrieved documents, helping ensure factual accuracy.

---

### 💬 Interactive UI

A Streamlit interface allows users to:

* Upload documents
* Build a knowledge base
* Ask questions
* View grounded responses

---

## 🏗️ System Architecture

```text
                User Query
                     │
                     ▼
              Input Guardrail
                     │
                     ▼
              Hybrid Retrieval
              ┌──────────────┐
              │              │
              ▼              ▼
        Dense Search     BM25 Search
              │              │
              └──────┬───────┘
                     ▼
         Reciprocal Rank Fusion
                     │
                     ▼
            Cross Encoder
               Reranker
                     │
                     ▼
             Context Builder
                     │
                     ▼
              LLM Generator
                     │
                     ▼
             Output Guardrail
                     │
                     ▼
             Grounded Answer
```

---

## 🧠 Retrieval Pipeline

### Document Processing

```text
Upload File
      │
      ▼
Document Loader
      │
      ▼
Chunking
      │
      ▼
Embedding Generation
      │
      ▼
Store in ChromaDB
      │
      ▼
Build BM25 Index
```

### Query Processing

```text
User Query
      │
      ▼
Dense Retrieval
      +
Sparse Retrieval
      │
      ▼
RRF Fusion
      │
      ▼
Cross Encoder Reranking
      │
      ▼
Context Construction
      │
      ▼
LLM Generation
      │
      ▼
Citation Validation
```

---

## ⚙️ Tech Stack

| Technology            | Purpose               |
| --------------------- | --------------------- |
| Python                | Core backend          |
| Streamlit             | User interface        |
| ChromaDB              | Vector database       |
| Sentence Transformers | Embeddings            |
| BM25                  | Sparse retrieval      |
| Cross Encoder         | Reranking             |
| Groq API              | LLM inference         |
| LangChain             | Document processing   |
| OCR                   | Image text extraction |

---

## 📂 Project Structure

```text
multimodal-rag/
│
├── embeddings/
├── generation/
├── guardrails/
├── ingestion/
├── models/
├── pipelines/
├── preprocessing/
├── retrieval/
├── sparse/
├── vectorstore/
│
├── streamlit_app.py
├── main.py
└── requirements.txt
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone <repository-url>

cd multimodal-rag
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

---

## ▶️ Run Application

```bash
streamlit run streamlit_app.py
```

---

## 🔬 Implemented Retrieval Techniques

* Dense Vector Retrieval
* BM25 Sparse Retrieval
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Reranking
* Hybrid Search
* Context Compression
* Citation Grounding

---

## 📈 Future Improvements

* Multi-document knowledge bases
* Query decomposition
* Agentic RAG workflows
* Knowledge graph retrieval
* Streaming responses
* Evaluation framework
* Retrieval observability
* Distributed vector storage

---

## 👨‍💻 Author

**Dhruva**

Built to explore modern retrieval systems, multimodal search, grounded generation, hybrid ranking strategies, and production-oriented RAG architectures.
