import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from datetime import datetime

role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
bucket = "telco-churn-prediction-bucket"

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"telco-training-{timestamp}"

sklearn_estimator = SKLearn(
    entry_point="train.py",              # this means: scripts/train.py
    source_dir="scripts",                # folder where train.py is
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    py_version="py3",
    dependencies=["requirements.txt"],   # also relative to source_dir
)
# ---------- Run Training Job ----------


sklearn_estimator.fit(job_name=job_name)

# ---------- 