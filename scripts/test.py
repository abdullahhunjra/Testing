import os
import boto3
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------- CONFIG ----------
BUCKET = "telco-churn-prediction-bucket"
MODEL_KEY = "telco-trained-models/LogisticRegression_selected.pkl"
FEATURES_KEY = "selected_k_best_features/LogisticRegression_selected_features.json"
PROC_PREFIX = "telco-processed-data/"
LOCAL_MODEL_PATH = "/tmp/LogisticRegression_selected.pkl"
LOCAL_FEATURE_PATH = "/tmp/LogisticRegression_selected_features.json"

# ---------- S3 Client ----------
s3 = boto3.client("s3")

# ---------- Download Model ----------
try:
    s3.download_file(BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
    print("✅ Model downloaded from S3.")
except Exception as e:
    print("❌ Error downloading model:", e)
    exit(1)

# ---------- Download Selected Features ----------
try:
    s3.download_file(BUCKET, FEATURES_KEY, LOCAL_FEATURE_PATH)
    print("✅ Selected features JSON downloaded.")
    with open(LOCAL_FEATURE_PATH) as f:
        selected_features = json.load(f)
except Exception as e:
    print("❌ Error downloading features JSON:", e)
    exit(1)

# ---------- Load Model ----------
model = joblib.load(LOCAL_MODEL_PATH)

# ---------- Load Test Data ----------
def load_csv(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj["Body"])

X_test = load_csv(PROC_PREFIX + "X_test.csv")
y_test = load_csv(PROC_PREFIX + "y_test.csv").values.ravel()

# ---------- Subset to Selected Features ----------
X_test = X_test[selected_features]

# ---------- Predict and Evaluate ----------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ---------- Print Evaluation ----------
print("\n📊 Evaluation Metrics for LogisticRegression_selected.pkl")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n🧾 Confusion Matrix:")
print(cm)

print("\n📄 Classification Report:")
print(report)
