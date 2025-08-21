

### === models.py ===
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
from joblib import load
from prophet import Prophet

# === Paths ===
XGB_MODEL_PATH = r"D:\LLM\agent\ml_models\muxgb_lead_model.json"
PROPHET_MODEL_PATH = r"D:\LLM\agent\ml_models\Revenue_ML_model.pkl"

# === Load Models ===
xgb_model = xgb.Booster()
xgb_model.load_model(XGB_MODEL_PATH)
prophet_model = load(PROPHET_MODEL_PATH)

# === Prophet Forecast Helper ===
def generate_forecast(periods=365):
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    return forecast

forecast_df = generate_forecast()
TRAIN_END_DATE = forecast_df['ds'].min().normalize()

# === Forecasting Logic ===
def get_forecast_by_date(ds: str):
    try:
        date_obj = pd.to_datetime(ds).normalize()
        if date_obj <= TRAIN_END_DATE:
            return {"error": f"Date {ds} is before or within training data. Select a future date beyond {TRAIN_END_DATE.date()}"}

        mask = (forecast_df['ds'] > TRAIN_END_DATE) & (forecast_df['ds'] <= date_obj)
        filtered = forecast_df.loc[mask]

        if filtered.empty:
            return {"error": f"Date {ds} is out of forecast range. Try a date within 365 days from training end."}

        cumulative = round(filtered['yhat'].sum(), 2)
        return {"date": ds, "cumulative_revenue_forecast": cumulative}

    except Exception as e:
        return {"error": f"Error parsing date: {str(e)}"}

# === Lead Conversion Logic ===
expected_columns = [
    'Promo Flag', 'Marketing Spend', 'Num Reps Active',
    'Call Count', 'Meeting Scheduled',
    'year', 'month', 'day',
    'Sales Channel_Online',
    'Seasonality_Q2', 'Seasonality_Q3', 'Seasonality_Q4',
    'Lead Source_LinkedIn', 'Lead Source_Referral', 'Lead Source_Web',
    'Job Role_Director', 'Job Role_Intern', 'Job Role_Manager',
    'Industry_Finance', 'Industry_Retail', 'Industry_Tech',
    'Region_North', 'Region_South', 'Region_West',
    'Product Interest_B', 'Product Interest_C',
]

class LeadData(BaseModel):
    Promo_Flag: int
    Marketing_Spend: float
    Num_Reps_Active: int
    Call_Count: int
    Meeting_Scheduled: int
    year: int
    month: int
    day: int
    Sales_Channel_Online: int = 0
    Seasonality_Q2: int = 0
    Seasonality_Q3: int = 0
    Seasonality_Q4: int = 0
    Lead_Source_LinkedIn: int = 0
    Lead_Source_Referral: int = 0
    Lead_Source_Web: int = 0
    Job_Role_Director: int = 0
    Job_Role_Intern: int = 0
    Job_Role_Manager: int = 0
    Industry_Finance: int = 0
    Industry_Retail: int = 0
    Industry_Tech: int = 0
    Region_North: int = 0
    Region_South: int = 0
    Region_West: int = 0
    Product_Interest_B: int = 0
    Product_Interest_C: int = 0

    def to_df(self):
        df = pd.DataFrame([self.dict()])
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        return df[expected_columns]

def predict_lead_conversion(input_dict: dict):
    try:
        lead = LeadData(**input_dict)
        df = lead.to_df()
        dmatrix = xgb.DMatrix(df)
        pred_prob = xgb_model.predict(dmatrix)[0]
        prediction = int(pred_prob >= 0.5)
        return {
            "conversion_probability": float(pred_prob),
            "likely_to_convert": bool(prediction)
        }
    except Exception as e:
        return {"error": str(e)}