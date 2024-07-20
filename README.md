# VANER: Biomedical Named Entity Recognition by LLM

## Introduction

To install the required packages, run the following command:

```bash
poetry install
```

## Llama Version

### Train

To train the model, run the following command:

```bash
poetry run python ./llama/unllama_train_vaner.py task llora_path llama_version
```

### Evaluate

To evaluate the model, run the following command:

```bash
poetry run python ./llama/unllama_evaluate_vaner.py task max_length kgtype align_mode llama_version
```

## Mistral Version

### Train

To train the model, run the following command:

```bash
poetry run python ./mistral/unmistral_train_vaner.py task lora_path
```

### Evaluate

To evaluate the model, run the following command:

```bash
poetry run python ./mistral/unmistral_evaluate_vaner.py task max_length kgtype align_mode
```
