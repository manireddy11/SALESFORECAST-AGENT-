import xgboost as xgb
import pickle

# Paths to your models
XGB_MODEL_PATH = r"D:\LLM\agent\ml_models\muxgb_lead_model.json"
SKLEARN_MODEL_PATH = r"D:\LLM\agent\FCS_ML_model.pkl"

# Load XGBoost model
def load_xgb_model(path=XGB_MODEL_PATH):
    model = xgb.Booster()
    model.load_model(path)
    return model

# Load scikit-learn model
def load_sklearn_model(path=SKLEARN_MODEL_PATH):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

# Save scikit-learn model
def save_sklearn_model(model, path=SKLEARN_MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(model, f)

# Example predict functions
def predict_xgb(model, dmatrix):
    return model.predict(dmatrix)

def predict_sklearn(model, X):
    return model.predict(X)