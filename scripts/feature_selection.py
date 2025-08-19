import os
import json
import boto3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------- CONFIG ----------
BUCKET = "telco-churn-prediction-bucket"
PROC_PREFIX = "telco-processed-data/"
FEATURE_PREFIX = "selected_k_best_features/"
PLOTS_PREFIX = "telco-feature-selection-plots/"

os.makedirs("/tmp", exist_ok=True)
s3 = boto3.client("s3")

# ---------- Load Data ----------
def load_csv(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(obj["Body"])

X_train = load_csv(PROC_PREFIX + "X_train.csv")
y_train = load_csv(PROC_PREFIX + "y_train.csv").values.ravel()

# ---------- Models ----------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

# ---------- Feature Selection -------------
for name, model in models.items():
    print(f"\n🔍 Evaluating features for: {name}")
    
    k_values = range(5, X_train.shape[1] + 1)
    best_score = 0
    best_k = 0
    best_feats = []
    mean_recalls = []

    for k in k_values:
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(X_train, y_train)
        score = cross_val_score(model, X_new, y_train, cv=5, scoring="recall").mean()
        mean_recalls.append(score)

        if score > best_score:
            best_score = score
            best_k = k
            best_feats = X_train.columns[selector.get_support()].tolist()

    # ---------- Save Selected Features (per model) ----------
    feature_file = f"/tmp/{name}_selected_features.json"
    s3_feature_key = FEATURE_PREFIX + f"{name}_selected_features.json"

    with open(feature_file, "w") as f:
        json.dump(best_feats, f, indent=4)

    s3.upload_file(feature_file, BUCKET, s3_feature_key)
    print(f"✅ Saved selected features for {name} to S3 → {s3_feature_key}")

    # ---------- Save Plot ----------
    plt.figure(figsize=(6, 4))
    plt.plot(list(k_values), mean_recalls, marker="o", color="blue")
    plt.title(f"{name} - Recall vs Number of Features")
    plt.xlabel("Number of Features")
    plt.ylabel("Recall (CV=5)")
    plt.grid(True)
    plt.tight_layout()

    plot_path = f"/tmp/{name}_feature_selection_plot.png"
    s3_plot_key = PLOTS_PREFIX + f"{name}_feature_selection_plot.png"
    plt.savefig(plot_path)
    s3.upload_file(plot_path, BUCKET, s3_plot_key)
    print(f"📊 Uploaded feature selection plot for {name} to S3 → {s3_plot_key}")

print("\n✅ All model feature selections completed and uploaded to S3.")

#