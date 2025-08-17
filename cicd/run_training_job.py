import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from datetime import datetime

role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"
bucket = "telco-churn-prediction-bucket"
entry_point_script = "scripts/train.py"

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
job_name = f"telco-training-{timestamp}"

sklearn_estimator = SKLearn(
    entry_point=entry_point_script,
    role=role,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    base_job_name="telco-train",
    source_dir=".",
    output_path=f"s3://{bucket}/telco-trained-models/",
    sagemaker_session=sagemaker.Session(),
)

sklearn_estimator.fit(job_name=job_name)
