# scripts/hpt_logistic_regression.py

import os
import json
import boto3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt

BUCKET = "telco-churn-prediction-bucket"
PROC_PREFIX = "telco-processed-data/"
ART_PREFIX = "telco-model-artifacts/"
RESULTS_PREFIX = "telco-hpt-results/"

# Load data from S3
s3 = boto3.client("s3")

def load_csv(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj["Body"])

X_train = load_csv(PROC_PREFIX + "X_train.csv")
y_train = load_csv(PROC_PREFIX + "y_train.csv").values.ravel()
X_test = load_csv(PROC_PREFIX + "X_test.csv")
y_test = load_csv(PROC_PREFIX + "y_test.csv").values.ravel()

# Load selected features
obj = s3.get_object(Bucket=BUCKET, Key=ART_PREFIX + "selected_features.json")
selected_features = json.loads(obj["Body"].read().decode("utf-8"))
features = selected_features  # Assuming it's a list

X_train = X_train[features]
X_test = X_test[features]

# Hyperparameters from SageMaker
penalty = os.environ.get("penalty", "l2")
C = float(os.environ.get("C", 1.0))
solver = os.environ.get("solver", "lbfgs")

print(f"🔧 Training LogisticRegression(penalty={penalty}, C={C}, solver={solver})")

model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluation
report = classification_report(y_test, preds, output_dict=True)
print("📊 Classification Report:")
print(json.dumps(report, indent=4))

# Save report to S3
os.makedirs("/tmp", exist_ok=True)
report_path = "/tmp/hpt_metrics.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)
s3.upload_file(report_path, BUCKET, RESULTS_PREFIX + "hpt_metrics.json")

# Plot scores
metrics = ["precision", "recall", "f1-score"]
values = [report["1"][m] for m in metrics]

plt.figure(figsize=(6, 4))
bars = plt.bar(metrics, values)
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.2f}", ha='center', va='bottom')

plt.title("Logistic Regression (Selected Features) - HPT Metrics")
plt.ylim(0, 1)
plt.tight_layout()
plot_path = "/tmp/hpt_plot.png"
plt.savefig(plot_path)
s3.upload_file(plot_path, BUCKET, RESULTS_PREFIX + "hpt_plot.png")

print("✅ HPT model trained, evaluated, metrics and plot saved to S3.")
