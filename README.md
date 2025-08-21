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




