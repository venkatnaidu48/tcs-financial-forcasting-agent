# TCS Financial Forecasting Agent

## Project Overview
This project implements a FastAPI service that acts as an AI agent to produce a **qualitative business outlook forecast** for TCS by:
- Fetching recent financial documents
- Extracting key financial metrics (Revenue, Net Profit, Op. Margin)
- Performing RAG-based semantic analysis of earnings call transcripts
- Synthesizing a structured JSON forecast with an LLM

Architecture: FastAPI -> Agent (LangChain + Tools) -> Tools (extractor + Qualitative RAG) -> MySQL logger

## Files & Structure
(see repository structure)

## Agent & Tool Design
### FinancialDataExtractorTool
- Uses `pdfplumber` to extract text from PDFs.
- Extracts metrics using regex heuristics first, with an optional LLM fallback to parse complex tables.

### QualitativeAnalysisTool
- Uses `sentence-transformers` to build embeddings and `faiss` for similarity search.
- Uses LLM to synthesize recurring themes and management sentiment.

### Master Prompt
A concise prompt instructs the LLM to synthesize the numeric and qualitative context into JSON fields (`financial_trends`, `management_outlook`, `risks`, `opportunities`, `forecast_summary`).

## Setup Instructions (exact)
1. Clone:
   ```bash
   git clone https://github.com/venkatnaidu4/tcs-financial-forecasting-agent.git
   cd tcs-financial-forecasting-agent


## Steps for execution
step1: extract the files
step2: create the python environment(venv)
step3: install the requirements.txt
step4: python -m uvicorn app.main:app --reload --port 8000
step5: http://127.0.0.1:8000 or http://127.0.0.1:8000/docs [docs add manually in the browser]
