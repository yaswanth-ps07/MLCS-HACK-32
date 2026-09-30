import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_data
from preprocess import prepare_data


MODEL_PATH = "models/random_forest.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def train_model():
    train_df, test_df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        train_df, test_df
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("\nModel saved:", MODEL_PATH)
    print("Preprocessor saved:", PREPROCESSOR_PATH)


if __name__ == "__main__":
    train_model()