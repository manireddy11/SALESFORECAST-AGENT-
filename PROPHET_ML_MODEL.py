# sales_forecast_pipeline.py

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
file_path = r"D:\LLM\datasets\synthetic_sales_leads_data.csv"
df = pd.read_csv(file_path)




# Ensure correct datetime format
df['ds'] = pd.to_datetime(df['ds'])

# Sort by date just in case
df = df.sort_values(by='ds').reset_index(drop=True)

# Prophet expects only ds (date) and y (target) columns
df_prophet = df[['ds', 'y']]

# Split the dataset
train_size = int(len(df) * 0.8)
val_size = int(len(df) * 0.1)
test_size = len(df) - train_size - val_size

train_df = df_prophet[:train_size]
val_df = df_prophet[train_size:train_size + val_size]
test_df = df_prophet[train_size + val_size:]

# Fit Prophet model on training data
model = Prophet()
model.fit(train_df)

# Forecast on validation period
val_future = val_df[['ds']].copy()
val_forecast = model.predict(val_future)

# Forecast on test period
test_future = test_df[['ds']].copy()
test_forecast = model.predict(test_future)

# Evaluation function
def evaluate(y_true, y_pred, label="Set"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"📊 {label} MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Evaluate Validation Set
evaluate(val_df['y'], val_forecast['yhat'], "Validation")

# Evaluate Test Set
evaluate(test_df['y'], test_forecast['yhat'], "Test")




# save_prophet_model.py

import pandas as pd
from prophet import Prophet
from joblib import dump
import os

# === Load Dataset ===
file_path = r"D:\LLM\datasets\synthetic_sales_leads_data.csv"
df = pd.read_csv(file_path)

# === Convert 'ds' to datetime and select Prophet columns ===
df['ds'] = pd.to_datetime(df['ds'])
df_prophet = df[['ds', 'y']]

# === Optional: Sort the data just in case ===
df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

# === Fit the Prophet Model ===
model = Prophet()
model.fit(df_prophet)

# === Save the model ===
save_path = r"D:\LLM\agent\Revenue_ML_model.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
dump(model, save_path)

print(f"✅ Model trained and saved to: {save_path}")



#############################################################
# forecast_next_year.py

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from joblib import load

# === Load Trained Model ===
model_path = r"D:\LLM\agent\Revenue_ML_model.pkl"
model = load(model_path)

# === Create Future DataFrame for Next 365 Days ===
future = model.make_future_dataframe(periods=365)

# === Predict Future Sales ===
forecast = model.predict(future)

# === Show Tail (last 365 days of predictions) ===
next_year_forecast = forecast.tail(365)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print("📅 Forecast for the Next 1 Year:")
print(next_year_forecast.head())

# === Optional: Plot forecast ===
fig1 = model.plot(forecast)
plt.title("📈 Sales Forecast for the Next 1 Year")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

# === Optional: Save forecast to CSV ===
output_csv = r"D:\LLM\forecast_outputs\one_year_forecast.csv"
next_year_forecast.to_csv(output_csv, index=False)
print(f"✅ Forecast saved to: {output_csv}")
