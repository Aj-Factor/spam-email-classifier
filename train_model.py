from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sms_spam.csv"
MODEL_PATH = ROOT / "spam_classifier.joblib"
METRICS_PATH = ROOT / "model_metrics.txt"

def main():
    data = pd.read_csv(DATA_PATH).dropna(subset=["label", "message"])
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"]) & (data["message"] != "")]
    if data["label"].nunique() != 2 or data["label"].value_counts().min() < 2:
        raise ValueError("Dataset must contain at least two examples of both ham and spam.")
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.25, random_state=42, stratify=data["label"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    joblib.dump(model, MODEL_PATH)
    result = (f"SPAM EMAIL CLASSIFIER - TEST RESULTS\nDataset rows: {len(data)}\n"
              f"Training rows: {len(X_train)}\nTesting rows: {len(X_test)}\n"
              f"Accuracy: {accuracy:.4f}\n\nClassification report:\n{report}\n"
              f"Confusion matrix (true/predicted; order ham, spam):\n{matrix}\n")
    METRICS_PATH.write_text(result, encoding="utf-8")
    print(result)
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
