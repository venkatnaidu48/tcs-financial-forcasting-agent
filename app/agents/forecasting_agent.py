import os
import json
import uuid
import re
from dotenv import load_dotenv
from openai import OpenAI

from ..tools.financial_data_extractor import extract_metrics_from_pdf
from ..tools.qualitative_rag_tool import QualitativeAnalysisTool, extract_themes_and_sentiment
from ..database.mysql_logger import log_request

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MASTER_PROMPT = """
You are a senior financial analyst.
Use the extracted metrics + management commentary to produce:
- financial_trends
- management_outlook
- risks
- opportunities
- forecast_summary
Return ONLY valid JSON.
"""

class ForecastingAgent:
    def __init__(self, rag=None):
        self.rag = rag if rag else QualitativeAnalysisTool()

    def generate_forecast(self, query, docs, transcripts):
        request_id = str(uuid.uuid4())

        extracted = []
        for d in docs:
            metrics, meta = extract_metrics_from_pdf(d["path"], d.get("url"))
            extracted.append({"source": d, "metrics": metrics, "meta": meta})

        self.rag.ingest_transcripts(transcripts)
        snippets = self.rag.query(query, 5)
        qual = extract_themes_and_sentiment(snippets)

        context = {
            "query": query,
            "metrics": extracted,
            "qualitative": qual
        }

        prompt = MASTER_PROMPT + "\n\nContext:\n" + json.dumps(context, indent=2)

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content
            match = re.search(r"(\{.*\})", text, flags=re.S)
            final_json = json.loads(match.group(1)) if match else {"forecast_summary": text}
        except Exception as e:
            print("OpenAI synthesis error:", e)
            final_json = {"forecast_summary": "Error generating forecast"}

        response = {
            "request_id": request_id,
            **final_json,
            "metadata": {"extracted": extracted, "qualitative": qual}
        }

        try:
            log_request(request_id, query, json.dumps(response))
        except:
            pass

        return response
