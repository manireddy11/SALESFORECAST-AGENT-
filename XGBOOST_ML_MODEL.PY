import pandas as pd
# Define the file path
file_path = r"D:\LLM\datasets\synthetic_sales_leads_data.csv"
df = pd.read_csv(file_path)
# 1. Print all column names
print("🔹 Column Names:")
print(df.columns.tolist())
print()

# 2. Print count of null (NaN) values in each column
print("🔹 Null Values per Column:")
print(df.isnull().sum())
print()

# 3. Print data types of each column
print("🔹 Data Types of Columns:")
print(df.dtypes)
print()

# Display the first 5 rows
print(df.head())
df.columns

print(df.isnull().sum())
# ...existing code...
print("🔹 Total NaN values in dataset:", df.isnull().sum().sum())



# Extract year, month, day from 'ds'
df['ds'] = pd.to_datetime(df['ds'])
df['year'] = df['ds'].dt.year
df['month'] = df['ds'].dt.month
df['day'] = df['ds'].dt.day


# One-hot encode and drop the first category to avoid multicollinearity
df = pd.get_dummies(
    df,
    columns=[
        'Sales Channel',
        'Seasonality',
        'Lead Source',
        'Job Role',
        'Industry',
        'Region',
        'Product Interest'
    ],
    drop_first=True
)

# Convert any boolean columns (result of one-hot encoding) to int (0 or 1)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Optional: Check data types
print(df.dtypes)


# Prepare input features and target for XGBoost
X = df.drop(columns=['ds', 'y', 'Converted'])  # input features (drop ds, y, Converted)
y = df['Converted'] 


# Full list of model input columns
columns_to_show = [
    'Promo Flag', 'Marketing Spend', 'Num Reps Active',
    'Call Count', 'Meeting Scheduled',
    'year', 'month', 'day',
    'Sales Channel_Online', 'Seasonality_Q2', 'Seasonality_Q3', 'Seasonality_Q4',
    'Lead Source_LinkedIn', 'Lead Source_Referral', 'Lead Source_Web',
    'Job Role_Director', 'Job Role_Intern', 'Job Role_Manager',
    'Industry_Finance', 'Industry_Retail', 'Industry_Tech',
    'Region_North', 'Region_South', 'Region_West',
    'Product Interest_B', 'Product Interest_C'
]

# Print columns in batches of 5
batch_size = 5
for i in range(0, len(columns_to_show), batch_size):
    batch = columns_to_show[i:i+batch_size]
    print(f"\n🔹 Columns {i+1} to {i+len(batch)}: {batch}")
    print(df[batch].head())



# Split into train, validation, and test sets (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("XGBoost train shape:", X_train.shape)
print("XGBoost validation shape:", X_val.shape)
print("XGBoost test shape:", X_test.shape)


# Define and train the XGBoost model
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define and train the XGBoost model
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0,
    scale_pos_weight=0.69,  # Handles class imbalance
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Predict and evaluate
y_pred = model.predict(X_test)

print("XGBoost validation shape:", X_val.shape)
print("XGBoost test shape:", X_test.shape)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
from sklearn.model_selection import cross_val_score


sample_input = pd.DataFrame([{
    'Promo Flag': 1,
    'Marketing Spend': 15000,
    'Num Reps Active': 5,
    'Call Count': 30,
    'Meeting Scheduled': 1,
    'year': 2025,
    'month': 7,
    'day': 27,
    'Sales Channel_Online': 1,
    'Seasonality_Q2': 0,
    'Seasonality_Q3': 1,
    'Seasonality_Q4': 0,
    'Lead Source_LinkedIn': 0,
    'Lead Source_Referral': 1,
    'Lead Source_Web': 0,
    'Job Role_Director': 0,
    'Job Role_Intern': 0,
    'Job Role_Manager': 1,
    'Industry_Finance': 0,
    'Industry_Retail': 0,
    'Industry_Tech': 1,
    'Region_North': 1,
    'Region_South': 0,
    'Region_West': 0,
    'Product Interest_B': 0,
    'Product Interest_C': 1,
}])

# Predict
pred = model.predict(sample_input)[0]
prob = model.predict_proba(sample_input)[0]
print("Manual Input Prediction =>", "Converted" if pred == 1 else "Not Converted")
print("Probabilities =>", prob)



from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier

# Assuming X and y are your features and target variables
# model = XGBClassifier()  ← use this if not defined earlier

f1 = make_scorer(f1_score)

f1_scores = cross_val_score(model, X, y, scoring=f1, cv=5)

print("F1 Scores from 5-fold CV:", f1_scores)
print("Average F1 Score (CV):", f1_scores.mean())


from joblib import dump

model_path = r"D:\SVD\zuxgb_lead_model.pkl"
dump(model, model_path)
print("✅ Model saved successfully to:", model_path)
