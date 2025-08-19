import os
import joblib
import json
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------- CONFIG ----------
BUCKET = "telco-churn-prediction-bucket"
PROC_PREFIX = "telco-processed-data/"
ART_PREFIX = "telco-model-artifacts/"
MODEL_PREFIX = "telco-trained-models/"
RESULTS_PREFIX = "telco-model-results/"
FEATURES_PREFIX = "selected_k_best_features/"

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

# ---------- Compute Class Weights ----------
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]

# ---------- Models ----------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight=class_weight_dict),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, class_weight=class_weight_dict),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight
    ),
}

results = {}
recall_scores = {}

# ---------- Training Loop ----------
for name, model in models.items():
    print(f"\n📦 Training model: {name}")

    # ---- All Features ----
    if name == "GradientBoosting":
        sample_weights = [class_weight_dict[i] for i in y_train]
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    results[name + "_all"] = report
    recall_scores[name + "_all"] = report["1"]["recall"]

    joblib.dump(model, f"/tmp/{name}_all.pkl")
    s3.upload_file(f"/tmp/{name}_all.pkl", BUCKET, f"{MODEL_PREFIX}{name}_all.pkl")

    # ---- Load Selected Features for This Model ----
    feature_key = f"{FEATURES_PREFIX}{name}_selected_features.json"
    print(f"🔍 Loading selected features for {name} from S3: {feature_key}")

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=feature_key)
        model_features = json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        print(f"⚠️ Feature file not found for {name}, skipping selected features training.")
        continue

    # ---- Selected Features ----
    if name == "GradientBoosting":
        model.fit(X_train[model_features], y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train[model_features], y_train)

    preds_sel = model.predict(X_test[model_features])
    report_sel = classification_report(y_test, preds_sel, output_dict=True)
    results[name + "_selected"] = report_sel
    recall_scores[name + "_selected"] = report_sel["1"]["recall"]

    joblib.dump(model, f"/tmp/{name}_selected.pkl")
    s3.upload_file(f"/tmp/{name}_selected.pkl", BUCKET, f"{MODEL_PREFIX}{name}_selected.pkl")

# ---------- Save All Results ----------
with open("/tmp/model_results.json", "w") as f:
    json.dump(results, f, indent=4)

s3.upload_file("/tmp/model_results.json", BUCKET, RESULTS_PREFIX + "model_results.json")

# ---------- Save All Reports as TXT ----------
report_txt_path = "/tmp/model_performance_report.txt"
with open(report_txt_path, "w") as f:
    for model_name, report_dict in results.items():
        f.write(f"===== {model_name} =====\n")
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                f.write(f"  {label}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"    {metric_name}: {value:.4f}\n")
            else:
                f.write(f"  {label}: {metrics:.4f}\n")
        f.write("\n\n")

s3.upload_file(report_txt_path, BUCKET, RESULTS_PREFIX + "model_performance_report.txt")

# ---------- Save Plot ----------
plt.figure(figsize=(10, 5))
bars = plt.bar(recall_scores.keys(), recall_scores.values(), color="skyblue", label="Recall Score")
plt.ylabel("Recall (Class = 1)")
plt.title("Model Recall Comparison - All vs Selected Features")
plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.005, f"{height:.3f}",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.legend(loc="upper right")
plt.tight_layout()
plot_path = "/tmp/recall_plot.png"
plt.savefig(plot_path)
s3.upload_file(plot_path, BUCKET, RESULTS_PREFIX + "recall_plot.png")

print("✅ All models trained, evaluated, and plotted. Saved to S3.")
