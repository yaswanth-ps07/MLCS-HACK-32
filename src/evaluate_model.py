import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data_loader import load_data
from preprocess import prepare_data


MODEL_PATH = "models/random_forest.pkl"


train_df, test_df = load_data()

X_train, X_test, y_train, y_test, _ = prepare_data(
    train_df, test_df
)

model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
)

disp.plot()
plt.title("UNSW-NB15 Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.show()

print("\nChart saved to: reports/confusion_matrix.png")