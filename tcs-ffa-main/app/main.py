from fastapi import FastAPI, HTTPException
from .schemas import ForecastRequest, ForecastResponse
from .agents.forecasting_agent import ForecastingAgent
from .tools.qualitative_rag_tool import QualitativeAnalysisTool
from .utils.scraper import fetch_screener_docs, download_file
import os

app = FastAPI(title="TCS Financial Forecasting Agent")


qual_tool = QualitativeAnalysisTool()
agent = ForecastingAgent(qual_tool)

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
   
    try:
        docs_meta = fetch_screener_docs()
    except Exception as e:
        docs_meta = []
    
    docs_to_download = docs_meta[:req.quarters]
    downloaded = []
    for d in docs_to_download:
        try:
            path = download_file(d["url"])
            downloaded.append({"title": d["title"], "path": path, "url": d["url"]})
        except Exception as e:
            
            continue

    transcripts = [
        {"title": "Earnings Call Q1", "text": "Management is focused on growth in digital services and operating margin discipline. We are cautious about wage inflation and macro."},
        {"title": "Earnings Call Q2", "text": "Strong deal wins in Europe. Management sees demand for cloud transformation. Potential margin pressure due to INR depreciation."},
        {"title": "Earnings Call Q3", "text": "Hiring ramp and investments will continue. Management expects moderate revenue growth next quarter."}
    ]

    response_dict = agent.generate_forecast(req.query, downloaded, transcripts)
    return response_dict
