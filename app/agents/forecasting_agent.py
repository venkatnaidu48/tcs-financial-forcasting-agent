
import uuid
import json
import os
from dotenv import load_dotenv
load_dotenv()

import openai
from ..tools.financial_data_extractor import extract_metrics_from_pdf
from ..tools.qualitative_rag_tool import QualitativeAnalysisTool, extract_themes_and_sentiment
from ..database.mysql_logger import log_request

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY


MASTER_PROMPT = """
You are a senior financial analyst. Use the numeric metrics and qualitative analysis
to produce a structured financial forecast.

Return ONLY valid JSON with keys:
- financial_trends
- management_outlook
- risks
- opportunities
- forecast_summary
"""


class ForecastingAgent:
    def __init__(self, qualitative_tool: QualitativeAnalysisTool = None):
        self.qual_tool = qualitative_tool if qualitative_tool else QualitativeAnalysisTool()

    def generate_forecast(self, query: str, docs: list, transcripts: list):
        request_id = str(uuid.uuid4())

        extracted_metrics = []
        for doc in docs:
            path = doc.get("path")
            metrics, meta = extract_metrics_from_pdf(path, source_url=doc.get("url"))
            extracted_metrics.append({
                "source": doc,
                "metrics": metrics,
                "meta": meta
            })

        self.qual_tool.ingest_transcripts(transcripts)

        qa_snippets = self.qual_tool.query(query, top_k=5)
        synthesis = extract_themes_and_sentiment(qa_snippets)

        context = {
            "query": query,
            "metrics": extracted_metrics,
            "qualitative": synthesis,
            "num_docs": len(docs),
            "num_transcripts": len(transcripts)
        }

        context_text = json.dumps(context, indent=2)

        final_output = {
            "financial_trends": {},
            "management_outlook": {},
            "risks": [],
            "opportunities": [],
            "forecast_summary": ""
        }

        if OPENAI_KEY:
            prompt = (
                MASTER_PROMPT
                + "\n\nContext:\n"
                + context_text
                + "\n\nReturn ONLY a JSON object."
            )

            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=900,
                )
                out = resp["choices"][0]["message"]["content"]

                match = re.search(r"(\{.*\})", out, flags=re.S)
                if match:
                    final_output = json.loads(match.group(1))
                else:
                    final_output["forecast_summary"] = out.strip()

            except Exception as e:
                print("OpenAI synthesis error:", e)
                final_output["forecast_summary"] = "Error generating forecast."

        else:
            final_output["forecast_summary"] = "No OpenAI key found; cannot generate forecast."

        response = {
            "request_id": request_id,
            "financial_trends": final_output.get("financial_trends", {}),
            "management_outlook": final_output.get("management_outlook", {}),
            "risks": final_output.get("risks", []),
            "opportunities": final_output.get("opportunities", []),
            "forecast_summary": final_output.get("forecast_summary", ""),
            "metadata": {
                "extracted_metrics": extracted_metrics,
                "qualitative": synthesis,
            },
        }

       
        try:
            log_request(request_id, query, json.dumps(response))
        except Exception as e:
            print("MySQL logging failed:", e)

        return response
