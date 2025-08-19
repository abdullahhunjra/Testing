import os
import json
import joblib
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------- CONFIG ----------
BUCKET = "telco-churn-prediction-bucket"
PROC_PREFIX = "telco-processed-data/"
RESULTS_PREFIX = "telco-hpt-results/"

os.makedirs("/tmp", exist_ok=True)
s3 = boto3.client("s3")

# ---------- Load Data ----------
def load_csv(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj["Body"])

X_train = load_csv(PROC_PREFIX + "X_train.csv")
y_train = load_csv(PROC_PREFIX + "y_train.csv").values.ravel()
X_test = load_csv(PROC_PREFIX + "X_test.csv")
y_test = load_csv(PROC_PREFIX + "y_test.csv").values.ravel()

# ---------- Hyperparameters from Environment ----------
penalty = os.environ.get("penalty", "l2")
C = float(os.environ.get("C", 1.0))
solver = os.environ.get("solver", "lbfgs")

print(f"🔧 Training LogisticRegression(penalty={penalty}, C={C}, solver={solver})")

# ---------- Train ----------
model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# ---------- Metrics ----------
report = classification_report(y_test, preds, output_dict=True)
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
cm = confusion_matrix(y_test, preds).tolist()

# ✅ REQUIRED: emit metric in exact format for SageMaker
print(f'"1" : {{"f1-score": {f1:.4f}}}')  # <- must match regex in runner

# ---------- Logs ----------
print("\n📊 Classification Report:")
print(json.dumps(report, indent=4))
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ Confusion Matrix: {cm}")

# ---------- Save Metrics JSON ----------
metrics_output = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "confusion_matrix": cm,
    "full_classification_report": report
}

with open("/tmp/hpt_metrics.json", "w") as f:
    json.dump(metrics_output, f, indent=4)

s3.upload_file("/tmp/hpt_metrics.json", BUCKET, RESULTS_PREFIX + "hpt_metrics.json")

# ---------- Save Metrics TXT ----------
with open("/tmp/hpt_metrics.txt", "w") as f:
    f.write("📊 Logistic Regression HPT Metrics (All Features)\n")
    f.write(f"Penalty: {penalty}, C: {C}, Solver: {solver}\n\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Confusion Matrix: {cm}\n\n")
    f.write(json.dumps(report, indent=4))

s3.upload_file("/tmp/hpt_metrics.txt", BUCKET, RESULTS_PREFIX + "hpt_metrics.txt")

# ---------- Save Model ----------
model_path = "/tmp/hpt_logistic_regression.pkl"
joblib.dump(model, model_path)
s3.upload_file(model_path, BUCKET, RESULTS_PREFIX + "hpt_logistic_regression.pkl")

# ---------- Save Bar Plot ----------
plt.figure(figsize=(6, 4))
bars = plt.bar(["Precision", "Recall", "F1 Score"], [prec, rec, f1], color="skyblue")
for bar, val in zip(bars, [prec, rec, f1]):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha='center', va='bottom')
plt.title("Logistic Regression (All Features) - HPT Metrics")
plt.ylim(0, 1.05)
plt.tight_layout()
bar_plot_path = "/tmp/hpt_plot.png"
plt.savefig(bar_plot_path)
s3.upload_file(bar_plot_path, BUCKET, RESULTS_PREFIX + "hpt_plot.png")

# ---------- Save Confusion Matrix Plot ----------
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_plot_path = "/tmp/hpt_confusion_matrix.png"
plt.savefig(cm_plot_path)
s3.upload_file(cm_plot_path, BUCKET, RESULTS_PREFIX + "hpt_confusion_matrix.png")

print("✅ Model trained and all outputs saved to S3.")
