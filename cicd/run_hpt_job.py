import os
from datetime import datetime
from sagemaker import Session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter

# ✅ Replace this with your actual SageMaker execution role ARN
role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"

# Initialize SageMaker session
session = Session()

# Dynamic job name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
job_name = f"telco-hpt-logreg-{timestamp}"

# SKLearn Estimator
estimator = SKLearn(
    entry_point="hpt_logistic_regression.py",
    source_dir="scripts",
    role=role,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    py_version="py3",
    dependencies=["requirements.txt"],
    base_job_name="telco-hpt-logreg",
    sagemaker_session=session
)

# Hyperparameter search space
hyperparameter_ranges = {
    "C": ContinuousParameter(0.01, 10.0),
    "penalty": CategoricalParameter(["l1", "l2"]),
    "solver": CategoricalParameter(["liblinear", "lbfgs", "saga"])
}

# Extract f1-score for class "1" from printed JSON
metric_definitions = [
    {
        "Name": "f1-score",
        "Regex": '"1"\\s*:\\s*{[^}]*"f1-score"\\s*:\\s*([0-9\\.]+)'
    }
]

# Hyperparameter tuner
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="f1-score",
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    max_jobs=10,
    max_parallel_jobs=2,
    objective_type="Maximize"
)

# Launch tuning job
tuner.fit(job_name=job_name)

print(f"✅ Launched HPT job: {job_name}")
