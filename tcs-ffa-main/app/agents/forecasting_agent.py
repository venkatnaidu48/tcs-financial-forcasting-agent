import uuid
from ..tools.financial_data_extractor import extract_metrics_from_pdf
from ..tools.qualitative_rag_tool import QualitativeAnalysisTool
from ..database.mysql_logger import log_request
from ..schemas import ForecastResponse
import json
import os
from langchain import OpenAI, LLMChain, PromptTemplate

OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)

MASTER_PROMPT = """
You are a financial analyst AI. You have access to:
1) numeric metrics: revenue/net_profit/operating_margin extracted from last N quarter reports
2) qualitative analysis results: themes, sentiment, forward-looking statements from earnings calls
Using those, synthesize a 4-part output:
- financial_trends: short bullets with reasoning and simple numbers (growth, yoy/qoq)
- management_outlook: summarize management tone
- risks: list top 4 risks
- opportunities: list top 4 opportunities
- forecast_summary: 2-3 sentence forecast for the next quarter
Return JSON with keys exactly: financial_trends, management_outlook, risks, opportunities, forecast_summary
"""

def _call_llm_for_synthesis(context_text):
    if not OPENAI_KEY:
        
        return {
            "financial_trends": {"summary": "Insufficient LLM key; show extracted metrics."},
            "management_outlook": {"summary": "LLM not configured."},
            "risks": [],
            "opportunities": [],
            "forecast_summary": "LLM unavailable - cannot synthesize forecast."
        }
    llm = OpenAI(temperature=0, model_name="gpt-4o-mini")
    template = PromptTemplate(input_variables=["context"], template=MASTER_PROMPT + "\n\nContext:\n{context}\n\nJSON:")
    chain = LLMChain(llm=llm, prompt=template)
    out = chain.run({"context": context_text[:12000]})
   
    import re, json
    m = re.search(r"(\{.*\})", out, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            return {"financial_trends": {}, "management_outlook": {}, "risks": [], "opportunities": [], "forecast_summary": out}
    return {"financial_trends": {}, "management_outlook": {}, "risks": [], "opportunities": [], "forecast_summary": out}

class ForecastingAgent:
    def __init__(self, qualitative_tool: QualitativeAnalysisTool):
        self.qual_tool = qualitative_tool

    def generate_forecast(self, query: str, docs: list, transcripts: list):
       
        request_id = str(uuid.uuid4())
      
        extracted_metrics = []
        for doc in docs:
            metrics, meta = extract_metrics_from_pdf(doc.get("path"), source_url=doc.get("url"))
            extracted_metrics.append({"metrics": metrics, "meta": meta, "source": doc})

        
        self.qual_tool.ingest_transcripts(transcripts)

        
        qa = self.qual_tool.analyze(query)

        
        context_chunks = []
        context_chunks.append("Extracted Metrics:\n" + json.dumps(extracted_metrics, indent=2))
        context_chunks.append("Qualitative Analysis:\n" + json.dumps(qa, indent=2))
        context = "\n\n".join(context_chunks)

        
        synthesis = _call_llm_for_synthesis(context)

        response = {
            "request_id": request_id,
            "financial_trends": synthesis.get("financial_trends", {}),
            "management_outlook": synthesis.get("management_outlook", {}),
            "risks": synthesis.get("risks", []),
            "opportunities": synthesis.get("opportunities", []),
            "forecast_summary": synthesis.get("forecast_summary", ""),
            "metadata": {
                "extracted_metrics": extracted_metrics,
                "qualitative_snippets_count": len(qa.get("snippets", [])),
            },
        }

        try:
            log_request(request_id, query, json.dumps(response))
        except Exception as e:
            print("DB log failed:", e)
        return response
