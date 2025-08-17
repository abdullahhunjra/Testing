# cicd/run_preprocessing_job.py
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import boto3

role = "arn:aws:iam::755283537318:role/telco-sagemaker-role"  # Replace with correct ARN
bucket = "telco-churn-prediction-bucket"

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",  # Change if needed
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="telco-preprocessing",
)

sklearn_processor.run(
    code="scripts/preprocess.py",
    inputs=[
        ProcessingInput(source=f"s3://{bucket}/telco-raw-data", destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/processed", destination=f"s3://{bucket}/telco-processed-data"),
        ProcessingOutput(source="/opt/ml/processing/artifacts", destination=f"s3://{bucket}/telco-model-artifacts"),
    ],
)
# Wait for the job to complete






