# app/schemas.py
from pydantic import BaseModel
from typing import Optional

class ForecastRequest(BaseModel):
    query: str
    quarters: Optional[int] = 3
    include_market_data: Optional[bool] = False

class ForecastResponse(BaseModel):
    request_id: str
    financial_trends: dict
    management_outlook: dict
    risks: list
    opportunities: list
    forecast_summary: str
    metadata: dict
