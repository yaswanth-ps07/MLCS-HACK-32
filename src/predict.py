import joblib
import pandas as pd

MODEL_PATH = "models/random_forest.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

CATEGORICAL_FEATURES = ["proto", "service", "state"]

DROP_COLUMNS = ["id", "attack_cat", "label"]


def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def predict_record(record):
    model, preprocessor = load_model()

    df = pd.DataFrame([record])

    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    X_processed = preprocessor.transform(X)

    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]

    result = "Attack" if prediction == 1 else "Normal"
    risk_level = get_risk_level(probability)

    return result, probability, risk_level

def get_risk_level(probability):
    if probability >= 0.70:
        return "High"
    elif probability >= 0.30:
        return "Medium"
    else:
        return "Low"