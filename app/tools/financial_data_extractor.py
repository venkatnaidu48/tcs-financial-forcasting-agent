import os
import re
import json
import pdfplumber
from dotenv import load_dotenv
from ..database.mysql_logger import save_document

from openai import OpenAI

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)


def extract_text_from_pdf(path):
    text_pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text_pages.append(p.extract_text() or "")
    except Exception as e:
        print("PDF error:", e)
        return ""
    return "\n".join(text_pages)


def simple_regex_extract(text):
    results = {}
    patterns = {
        "total_revenue": r"(?:total\s+revenue|revenue)[^\d]{0,40}([\d,\.]+\s*(?:crore|cr|₹|rs|INR)?)",
        "net_profit": r"(?:net\s+profit|pat)[^\d]{0,40}([\d,\.]+\s*(?:crore|cr|₹|rs|INR)?)",
        "operating_margin": r"(?:operating\s+margin)[^\d]{0,40}([\d\.]+%)"
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, flags=re.I)
        if m:
            results[k] = m.group(1).strip()
    return results


def llm_extract_summary(text):
    if not OPENAI_KEY:
        return {}

    prompt = (
        "Extract Total Revenue, Net Profit, Operating Margin as JSON.\n"
        "Keys: total_revenue, net_profit, operating_margin.\n\n"
        f"TEXT:\n{text[:15000]}\n\nJSON:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        content = response.choices[0].message.content
        match = re.search(r"(\{.*\})", content, flags=re.S)
        if match:
            return json.loads(match.group(1))
    except Exception as e:
        print("LLM extraction error:", e)

    return {}


def extract_metrics_from_pdf(path, source_url=None):
    text = extract_text_from_pdf(path)

    if source_url:
        try:
            save_document("pdf", source_url, os.path.basename(path), text)
        except Exception as e:
            print("Save document error:", e)

    heur = simple_regex_extract(text)
    if len(heur) >= 2:
        return heur, {"method": "regex"}

    llm_res = llm_extract_summary(text)
    if llm_res:
        return llm_res, {"method": "llm"}

    return {}, {"method": "none"}
