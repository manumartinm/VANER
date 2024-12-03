from sagemaker.huggingface import HuggingFace

# Define HuggingFace estimator for distributed training
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    instance_type="ml.c5.xlarge",
    instance_count=2,
    role="role",
    transformers_version="4.17",
    pytorch_version="1.10",
    py_version="py38",
    hyperparameters={
        "epochs": 20,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "lora_r": 12,
    },
    distribution={
        "smdistributed": {
            "dataparallel": {
                "enabled": True
            }
        }
    }
)

huggingface_estimator.fit({"train": "s3://vrain/vaner/llama/data/train", "test": "s3://vrain/vaner/llama/data/test"})
