# TCS Financial Forecasting Agent

## Project Overview
This project implements a FastAPI service that acts as an AI agent to produce a **qualitative business outlook forecast** for TCS by:
- Fetching recent financial documents
- Extracting key financial metrics (Revenue, Net Profit, Op. Margin)
- Performing RAG-based semantic analysis of earnings call transcripts
- Synthesizing a structured JSON forecast with an LLM

Architecture: FastAPI -> Agent (LangChain + Tools) -> Tools (extractor + Qualitative RAG) -> MySQL logger

## Files & Structure
app/main.py — FastAPI server

agents/ — Forecasting agent logic

tools/ — Data extraction + qualitative RAG tools

utils/ — Scraper + embeddings

database/ — MySQL logging

sql/create_tables.sql — Required DB schema

## Agent & Tool Design
### FinancialDataExtractorTool
-Extracts text from PDFs using pdfplumber
Uses regex to detect key metrics
If regex fails, uses OpenAI 1.x API to extract values

### QualitativeAnalysisTool
-Creates embeddings using sentence-transformers
Stores and searches vectors using ChromaDB (replaces FAISS)
Uses OpenAI to generate qualitative summaries:
recurring themes
risk factors
management sentiment

### Master Prompt
The LLM synthesizes both numeric + qualitative data into JSON fields:
financial_trends
management_outlook
risks
opportunities
forecast_summary
confidence_score

## Setup Instructions (exact)
1. Clone:
   ```bash
   git clone https://github.com/venkatnaidu4/tcs-financial-forecasting-agent.git
   cd tcs-financial-forecasting-agent


## Steps for execution

Step 1: Extract/download project files
Step 2: Create and activate Python virtual environment
Step 3: Install dependencies (requirements.txt)
Step 4: Start FastAPI server:

## Open in browser:

http://127.0.0.1:8000

http://127.0.0.1:8000/docs

http://127.0.0.1:8000/forecast
(or)

## Steps for execution
step1: extract the files
step2: create the python environment(venv)
step3: install the requirements.txt
step4: python -m uvicorn app.main:app --reload --port 8000
step5: http://127.0.0.1:8000 or http://127.0.0.1:8000/docs [docs add manually in the browser] or http://127.0.0.1:8000/forecast [docs add manually in the browser]