import os
import re
import json
import pdfplumber
from dotenv import load_dotenv

from ..database.mysql_logger import save_document

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

import openai
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY


def extract_text_from_pdf(path):
    """Extract all text from a PDF file as a string."""
    text_pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text() or ""
                text_pages.append(page_text)
    except Exception as e:
        print("Error reading PDF:", e)
        return ""

    return "\n".join(text_pages)


def simple_regex_extract(text):
    """Heuristic extraction of revenue, net profit, operating margin."""
    results = {}

    patterns = {
        "total_revenue": r"(?:total\s+revenue|revenue|net\s+revenue)[^\d\n\r]{0,40}([\d,\.]+\s*(?:crore|cr|₹|rs|Rs|INR)?)",
        "net_profit": r"(?:net\s+profit|profit\s+after\s+tax|pat)[^\d\n\r]{0,40}([\d,\.]+\s*(?:crore|cr|₹|rs|Rs|INR)?)",
        "operating_margin": r"(?:operating\s+margin|op\.?\s*margin|ebitda\s+margin)[^\d\n\r]{0,40}([\d\.]+%)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            results[key] = m.group(1).strip()

    return results


def llm_extract_summary(text):
    """Use OpenAI model to extract financial metrics as JSON."""
    if not OPENAI_KEY:
        return {}

    prompt = (
        "You are a financial data extractor.\n"
        "From the quarterly report text below, extract:\n"
        "- Total Revenue\n"
        "- Net Profit (PAT)\n"
        "- Operating Margin\n\n"
        "Return ONLY valid JSON with keys:\n"
        "total_revenue, net_profit, operating_margin.\n"
        "Use null for missing values.\n\n"
        f"TEXT:\n{text[:15000]}\n\nJSON:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        out = response["choices"][0]["message"]["content"]

        match = re.search(r"(\{.*\})", out, flags=re.S)
        if match:
            return json.loads(match.group(1))

    except Exception as e:
        print("OpenAI LLM extraction error:", e)

    return {}


def extract_metrics_from_pdf(path, source_url=None):
    """Main function: PDF → text → regex → LLM fallback."""
    text = extract_text_from_pdf(path)


    if source_url:
        try:
            save_document("pdf", source_url, os.path.basename(path), text)
        except Exception as e:
            print("Document save error:", e)


    heuristics = simple_regex_extract(text)
    if len(heuristics) >= 2:
        return heuristics, {"extraction_method": "regex", "path": path}

    llm_result = llm_extract_summary(text)
    if llm_result:
        return llm_result, {"extraction_method": "llm", "path": path}

    return {}, {"extraction_method": "none", "path": path}
