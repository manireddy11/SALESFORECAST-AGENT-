

### === main.py ===
from fastapi import FastAPI, HTTPException
from models import get_forecast_by_date, predict_lead_conversion, LeadData

app = FastAPI(title="ML Prediction API")

@app.get("/forecast/{date}")
def api_get_forecast(date: str):
    result = get_forecast_by_date(date)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/lead_conversion/")
def api_lead_conversion(lead: LeadData):
    result = predict_lead_conversion(lead.dict())
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/")
def root():
    return {"message": "API is working. Use /forecast/{date} or /lead_conversion/ endpoints."}






