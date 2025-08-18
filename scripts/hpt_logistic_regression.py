import os
import json
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
FEATURES_PREFIX = "selected_k_best_features/"
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

# ---------- Load Selected Features for Logistic Regression ----------
feature_key = f"{FEATURES_PREFIX}LogisticRegression_selected_features.json"
try:
    obj = s3.get_object(Bucket=BUCKET, Key=feature_key)
    features = json.loads(obj["Body"].read().decode("utf-8"))
except s3.exceptions.NoSuchKey:
    raise FileNotFoundError(f"❌ Selected feature file not found at {feature_key}")

# Subset features
X_train = X_train[features]
X_test = X_test[features]

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
cm = confusion_matrix(y_test, preds).tolist()  # convert numpy to list for JSON

# Log in CloudWatch
print("\n📊 Classification Report:")
print(json.dumps(report, indent=4))
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ Confusion Matrix: {cm}")

# ---------- Save Results ----------
metrics_output = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "confusion_matrix": cm,
    "full_classification_report": report
}

# Save metrics JSON
metrics_path = "/tmp/hpt_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_output, f, indent=4)
s3.upload_file(metrics_path, BUCKET, RESULTS_PREFIX + "hpt_metrics.json")

# ---------- Save Bar Plot (Precision, Recall, F1) ----------
plt.figure(figsize=(6, 4))
bars = plt.bar(["Precision", "Recall", "F1 Score"], [prec, rec, f1], color="skyblue")
for bar, val in zip(bars, [prec, rec, f1]):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha='center', va='bottom')
plt.title("Logistic Regression (Selected Features) - HPT Metrics (Class 1)")
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
