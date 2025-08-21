# SALESFORECAST-AGENT-




🚀 Sales Forecasting AI Agent

An AI-powered Sales Forecasting & Lead Conversion Agent that predicts future revenue pipelines and lead conversion probabilities by calling backend machine learning models trained on the business data of X Company.

This project combines the power of LangChain, Mistral LLM, and custom ML models (XGBoost & Prophet) to deliver actionable insights for business growth.

📌 Project Description

This project focuses on predictive sales intelligence by integrating:

✅ Lead Conversion Prediction using an XGBoost model (muxgb_lead_model.json)

✅ Revenue Forecasting using a Facebook Prophet model (Revenue_ML_model.pkl)

Both models are saved locally and invoked through LangChain tool functions, ensuring predictions are returned seamlessly even though large language models like Mistral cannot directly integrate with ML models.

👉 LangChain acts as the bridge, enabling LLMs to communicate with ML models and return domain-specific predictions.

⚡ Features

🔮 Sales Revenue Forecasting: Predict cumulative sales pipeline revenue up to any given future date.

📊 Lead Conversion Probability: Estimate the likelihood of lead conversion based on historical data.

🔗 LangChain Tool Integration: Calls ML models via tool functions rather than relying solely on the LLM.

🖥️ Streamlit Interface: User-friendly frontend that ensures model input columns align with trained model expectations.

🤖 Mistral LLM Powered Agent: Orchestrates conversation and reasoning while delegating prediction tasks to ML models.

![App screenshot](https://raw.githubusercontent.com/manireddy11/SALESFORECAST-AGENT-/c0ae939445cf3d45669dd281b75e7664a35434bf/Screenshot%202025-08-21%20145731.png)


🚧 Challenges Faced

Choosing the Right LLM

Initially attempted with LLaMA but ran into CMake dependency issues with VS Code and version mismatches.

Solved by switching to Mistral, which integrates smoothly with LangChain and avoids heavy dependencies.

Bridging LLMs with ML Models

Open-source LLMs (unlike OpenAI, Claude, Gemini) cannot directly consume ML model outputs.

Required a LangChain tool function approach to handle ML predictions.

Data Schema Alignment

ML models expect specific input columns.

Without exact alignment, predictions fail.

Solution: enforced strict column validation in Streamlit UI to match model expectations.

✅ How I Overcame These Challenges

🚀 Used LangChain tool functions to bypass direct LLM-ML integration.

⚡ Enforced column validation in the frontend to ensure smooth predictions.

🔄 Migrated from LLaMA to Mistral, eliminating dependency issues.

🎯 Leveraged domain-specific models for business-focused AI agent development.

🛠️ Tech Stack

Backend Models:

XGBoost
 – Lead Conversion Prediction

Facebook Prophet
 – Revenue Forecasting

Agent & Orchestration:

LangChain
 – Tool functions & agent logic

Mistral LLM
 – Natural language reasoning

Frontend:

Streamlit
 – Interactive UI for predictions

Deployment:

FastAPI
 – Backend service endpoints

Uvicorn
 – ASGI server

[IMAGE ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::]

🧪 API Testing with Postman

The backend models were thoroughly tested using Postman to validate predictions with the expected input schema.

🔹 Prophet Model (Revenue Forecasting)

Input: date (in dd-mm-yyyy format)

Output: Cumulative revenue forecast from the model’s training cutoff date up to the given date.

✔ Example:

{
  "date": "15-09-2027"
}


[IMAGE1/POSTMAN:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::]

 🔹 XGBoost Model (Lead Conversion Prediction)

This model requires specific feature columns exactly as trained.

Expected Input Columns:

{
  "Promo_Flag": 1,
  "Marketing_Spend": 5000,
  "Num_Reps_Active": 12,
  "Call_Count": 30,
  "Meeting_Scheduled": 4,
  "year": 2025,
  "month": 8,
  "day": 21,
  "Sales_Channel_Online": 1
}


Output: Probability of lead conversion (between 0 and 1).
[IMAGE 2 /POSTMAN :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::]
👉 Why this matters:

The ML models are schema-sensitive. If input columns don’t match exactly, predictions will fail.

To solve this, the Streamlit interface enforces correct input formats, and Postman testing validates the endpoints independently.






🚀 Getting Started QUICK GUIDE 
🔧 Installation
git clone https://github.com/your-username/sales-forecasting-agent.git
cd sales-forecasting-agent
pip install -r requirements.txt


▶️ Running the App

Start FastAPI backend

uvicorn main:app --reload


Run Streamlit frontend

streamlit run app.py

📌 Example Workflow

Enter future date → get cumulative sales forecast

Input lead features → get conversion probability

LLM agent (Mistral) handles user queries, while LangChain tool functions invoke ML models behind the scenes.
