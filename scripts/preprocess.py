import os, joblib, json, boto3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from io import StringIO

# ---------- CONFIG ----------
BUCKET = "telco-churn-prediction-bucket"
RAW_KEY = "telco-raw-data/Telco-Customer-Churn.csv"
PROC_PREFIX = "telco-processed-data/"
ART_PREFIX = "telco-model-artifacts/"

s3 = boto3.client("s3")

# 1. Download raw data from S3
print("Downloading raw data from S3...")
obj = s3.get_object(Bucket=BUCKET, Key=RAW_KEY)
df = pd.read_csv(obj["Body"])

# 2. Drop customerID & duplicates
df = df.drop("customerID", axis=1, errors="ignore").drop_duplicates()

# 3. Convert TotalCharges to numeric & drop nulls
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# Define cols
cat_cols = [
    "gender","Partner","Dependents","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod"
]
num_cols = ["tenure","MonthlyCharges","TotalCharges","SeniorCitizen"]

# 4. Split train/test
X, y = df.drop("Churn", axis=1), df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Balance training set
train_df = pd.concat([X_train, y_train], axis=1)
majority = train_df[train_df.Churn == "No"]
minority = train_df[train_df.Churn == "Yes"]
majority_downsampled = resample(
    majority, replace=False, n_samples=len(minority), random_state=42
)
train_balanced = pd.concat([majority_downsampled, minority])
X_train, y_train = train_balanced.drop("Churn", axis=1), train_balanced["Churn"]

# 6. Encode categorical
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(X_train[col])
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le

y_train = y_train.map({"No": 0, "Yes": 1})
y_test = y_test.map({"No": 0, "Yes": 1})

# 7. Scale numeric
scaler_index = {}
for col in num_cols:
    sc = StandardScaler()
    X_train[col] = sc.fit_transform(X_train[[col]])
    X_test[col] = sc.transform(X_test[[col]])
    joblib.dump(sc, f"/tmp/{col}_scaler.pkl")
    s3.upload_file(f"/tmp/{col}_scaler.pkl", BUCKET, ART_PREFIX + f"{col}_scaler.pkl")
    scaler_index[col] = f"{col}_scaler.pkl"

# 8. Feature Selection (SelectKBest)
selector = SelectKBest(score_func=mutual_info_classif, k=10)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.get_support()].tolist()

with open("/tmp/selected_features.json", "w") as f:
    json.dump(selected_features, f)
s3.upload_file("/tmp/selected_features.json", BUCKET, ART_PREFIX + "selected_features.json")

print("Selected features:", selected_features)

# 9. Save processed splits to CSV (upload to S3)
def upload_df(df, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=csv_buffer.getvalue())

upload_df(X_train, PROC_PREFIX + "X_train.csv")
upload_df(y_train.to_frame(), PROC_PREFIX + "y_train.csv")
upload_df(X_test, PROC_PREFIX + "X_test.csv")
upload_df(y_test.to_frame(), PROC_PREFIX + "y_test.csv")

# 10. Save encoders + metadata
joblib.dump(encoders, "/tmp/label_encoders.pkl")
s3.upload_file("/tmp/label_encoders.pkl", BUCKET, ART_PREFIX + "label_encoders.pkl")

with open("/tmp/scaler_index.json", "w") as f:
    json.dump(scaler_index, f)
s3.upload_file("/tmp/scaler_index.json", BUCKET, ART_PREFIX + "scaler_index.json")

print("Preprocessing complete. Data, artifacts, and selected features uploaded to S3.")
