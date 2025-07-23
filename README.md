# 🧠 Scientific QA Pipeline

A modular pipeline designed to help users ask natural language questions and receive answers based on scientific research papers from **arXiv**.

---

## 🚀 Project Goals

This project enables:
- **Scraping scientific papers** from arXiv using a keyword/topic.
- **Storing metadata**: title, summary, authors, publication date, PDF URL, and file path.
- **Preprocessing papers** into Markdown via [**Docling**](https://github.com/docling-project/docling) for downstream NLP.
- **Answering user queries** interactively using a LangGraph-powered multi-tool chat engine.

---

## 🗂️ Features

### ✅ 1. arXiv Scraper

- Located in `arxiv_scraper.py`
- Retrieves up to `max_results` papers per topic
- Downloads the PDF
- Extracts and saves:
  - `title`
  - `pdf_url`
  - `filepath`
  - `summary`
  - `authors`
  - `published date`
- Stores all metadata in `pdfs_metadata.json`

### 📄 2. PDF Preprocessing with Docling

- Converts PDFs into structured **Markdown**
- Improves readability and semantic structure
- Markdown used for:
  - Context parsing
  - Subquery generation
  - Retrieval-friendly chunking

---

## 💬 3. Conversational QA with LangGraph

[LangGraph](https://langchain-ai.github.io/langgraph/) is used to power the entire **stateful QA workflow**.

### 📌 Why LangGraph?

- **Graph-style orchestration** of steps
- **State management** across tools
- **Monitoring**: logs start and end of every tool
- **Prompt Saving**: pulled and optionally pushed via LangSmith
- **Fallback Handling**: retries with local prompts if remote fails

### 🧩 Integrated Tools

The system uses purpose-built tools for scientific document reasoning:
- `topic_extractor_tool`
- `subquery_generator_tool`
- `query_qdrant_tool`
- `tavily_search_tool`
- `answer_generator_tool`

Each tool plays a role in understanding and answering user questions grounded in scientific texts.

---

### 🔐 Environment Variables

To run the pipeline successfully, make sure the following keys are provided in a `.env` file or set in your environment:

```env
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your_qdrant_instance_url
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=scientific-qa
TAVILY_API_KEY=your_tavily_api_key
```
📝 Note: These credentials are required for:  
- Qdrant: embedding storage and retrieval
- LangSmith: prompt management, trace logging
- Tavily: fallback web search