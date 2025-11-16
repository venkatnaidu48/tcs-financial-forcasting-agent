# app/tools/financial_data_extractor.py
import pdfplumber
import re
from langchain import OpenAI, LLMChain, PromptTemplate
import os
from ..database.mysql_logger import save_document

OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)

def extract_text_from_pdf(path):
    text_pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text_pages.append(p.extract_text() or "")
    return "\n".join(text_pages)

def simple_regex_extract(text):
    """
    Try to find metrics heuristically:
    - Revenue
    - Net Profit
    - Operating Margin
    """
    results = {}
   
    patterns = {
        "total_revenue": r"(?:total\s+revenue|revenue|net\s+revenue)[^\d\n\r]{0,30}([\d,\.]+\s*(?:crore|cr|₹|rs|Rs|INR)?)",
        "net_profit": r"(?:net\s+profit|profit\s+after\s+tax|pat)[^\d\n\r]{0,30}([\d,\.]+\s*(?:crore|cr|₹|rs|Rs|INR)?)",
        "operating_margin": r"(?:operating\s+margin|op\.?\s*margin|ebitda\s+margin)[^\d\n\r]{0,30}([\d\.]+%)"
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            results[k] = m.group(1).strip()
    return results

def llm_extract_summary(text):
    if not OPENAI_KEY:
        return {}
    llm = OpenAI(temperature=0, model_name="gpt-4o-mini")  # replace with available model
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are a financial data extractor. From the following company quarterly report text, "
            "extract Total Revenue, Net Profit, Operating Margin and return JSON with keys: "
            "total_revenue, net_profit, operating_margin. If missing, set null.\n\nTEXT:\n{text}\n\nJSON:"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    out = chain.run({"text": text[:10000]})
    import json, re
    jmatch = re.search(r"(\{.*\})", out, flags=re.S)
    if jmatch:
        try:
            return json.loads(jmatch.group(1))
        except:
            return {}
    return {}

def extract_metrics_from_pdf(path, source_url=None):
    text = extract_text_from_pdf(path)
    if source_url:
        save_document("pdf", source_url, path.split("/")[-1], text)
    heur = simple_regex_extract(text)
    if len(heur) >= 2:
        return heur, {"extraction_method": "regex"}
    llm_result = llm_extract_summary(text)
    return llm_result, {"extraction_method": "llm" if llm_result else "none"}
