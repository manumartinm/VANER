# -*- coding: utf-8 -*-

import json
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import PeftModel
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List

from vaner.llama.modeling_llama import UnmaskingLlamaForTokenClassification
from vaner.utils_vaner import *
from vaner.seqeval.metrics.sequence_labeling import (
    f1_score,
    precision_score,
    recall_score,
)


def calculate_accuracy(y_true, y_pred):
    total_correct = 0
    total_samples = 0

    for true_sequence, pred_sequence in zip(y_true, y_pred):
        total_samples += len(true_sequence)
        total_correct += sum(
            1
            for true_label, pred_label in zip(true_sequence, pred_sequence)
            if true_label == pred_label
        )

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy


def load_ncbi_test(dname):
    data = []
    with open(f"./data/vaner_datacohort/ncbi/test_df_mix2.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "ncbi", "Disease")
            data.extend(items)
    return data


def load_ncbi():
    ret = {}
    train_data = load_ncbi_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_ncbi_test("test"))
    return DatasetDict(ret)


def load_BC2GM_test(kg_type):
    data = []
    with open(f"./data/vaner_datacohort/BC2GM/test_df_mix2.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "BC2GM", "Gene")
            data.extend(items)
    return data


def load_BC2GM():
    ret = {}
    train_data = load_BC2GM_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_BC2GM_test("test"))
    return DatasetDict(ret)


def load_JNLPBA_test(dname):
    data = []
    with open(f"./data/vaner_datacohort/JNLPBA/test_df_mix2.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "JNLPBA", "Gene")
            data.extend(items)
    return data


def load_JNLPBA():
    ret = {}
    train_data = load_JNLPBA_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_JNLPBA_test("test"))
    return DatasetDict(ret)


def load_linnaeus_test(dname):
    data = []
    with open(f"./data/vaner_datacohort/linnaeus/test_df_mix2.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "linnaeus", "Species")
            data.extend(items)
    return data


def load_linnaeus():
    ret = {}
    train_data = load_linnaeus_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_linnaeus_test("test"))
    return DatasetDict(ret)


def load_BC5CDR_chem_test(dname):
    data = []
    with open(f"./data/vaner_datacohort/BC5CDR-chem/test_df_mix2.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "BC5CDR-chem", "Chemical")
            data.extend(items)
    return data


def load_BC5CDR_chem():
    ret = {}
    train_data = load_BC5CDR_chem_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_BC5CDR_chem_test("test"))
    return DatasetDict(ret)


def load_BC5CDR_disease_test(dname):
    data = []
    with open(
        f"./data/vaner_datacohort/BC5CDR-disease/test_df_mix2.jsonl", "r"
    ) as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "BC5CDR-disease", "Disease")
            data.extend(items)
    return data


def load_BC5CDR_disease():
    ret = {}
    train_data = load_BC5CDR_disease_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_BC5CDR_disease_test("test"))
    return DatasetDict(ret)


def load_BC4CHEMD_test(dname):
    data = []
    with open(f"./data/vaner_datacohort/BC4CHEMD/test_df_mix2.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "BC4CHEMD", "Chemical")
            data.extend(items)
    return data


def load_BC4CHEMD():
    ret = {}
    train_data = load_BC4CHEMD_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_BC4CHEMD_test("test"))
    return DatasetDict(ret)


def load_s800_test(dname):
    data = []
    with open(f"./data/vaner_datacohort/s800/test_df_mix2.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "s800", "Species")
            data.extend(items)
    return data


def load_s800():
    ret = {}
    train_data = load_s800_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_s800_test("test"))
    return DatasetDict(ret)


def load_craft_chemicals_test(dname):
    data = []
    with open(f"./data/craft/test_craft_chemicals_prompt.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "craft_chemicals", "Chemical")
            data.extend(items)
    return data


def load_craft_chemicals():
    ret = {}
    train_data = load_craft_chemicals_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_craft_chemicals_test("test"))
    return DatasetDict(ret)


def load_craft_genes_test(dname):
    data = []
    with open(f"./data/craft/test_craft_genes_prompt.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "test_craft_genes", "Gene")
            data.extend(items)
    return data


def load_craft_genes():
    ret = {}
    train_data = load_craft_genes_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_craft_genes_test("test"))
    return DatasetDict(ret)


def load_craft_species_test(dname):
    data = []
    with open(f"./data/craft/test_craft_species_prompt.jsonl", "r") as reader:
        for line in reader:
            items = parse_mt(json.loads(line), "test", "craft_species", "Species")
            data.extend(items)
    return data


def load_craft_species():
    ret = {}
    train_data = load_craft_species_test("test")
    ret["train"] = Dataset.from_list(train_data)
    ret["test"] = Dataset.from_list(load_craft_species_test("test"))
    return DatasetDict(ret)


def tokenize_and_align_labels_new(examples, tokenizer, label2id, max_length=128):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="longest",
        max_length=max_length,
        truncation=True,
    )
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        prompt_len = 1

        word_ids_adjust = []
        for word_idx in word_ids:
            if word_idx is None:
                word_ids_adjust.append(None)
            elif word_idx == 0:
                word_ids_adjust.append(0)
            else:
                word_ids_adjust.append(word_idx - 1)

        for word_idx in word_ids_adjust:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < prompt_len:
                    label_ids.append(-100)
                else:
                    label_ids.append(label2id.get(label[word_idx - prompt_len], -100))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = calculate_accuracy(
        [item for sublist in true_labels for item in sublist],
        [item for sublist in true_predictions for item in sublist],
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def cross_validate(
    task: str,
    lora_path: str,
    llama_version: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    lora_r: int,
    n_splits: int = 5,
):
    if task == "ncbi":
        ds_full = load_ncbi()["train"]
        label2id = {"O": 0, "B-disease": 1, "I-disease": 2}
    elif task == "s800":
        ds_full = load_s800()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "linnaeus":
        ds_full = load_linnaeus()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "JNLPBA":
        ds_full = load_JNLPBA()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "BC5CDR-chem":
        ds_full = load_BC5CDR_chem()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "BC5CDR-disease":
        ds_full = load_BC5CDR_disease()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "BC4CHEMD":
        ds_full = load_BC4CHEMD()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "BC2GM":
        ds_full = load_BC2GM()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "craft_chemicals":
        ds_full = load_craft_chemicals()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "craft_genes":
        ds_full = load_craft_genes()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    elif task == "craft_species":
        ds_full = load_craft_species()["train"]
        label2id = {"O": 0, "B-biomedical": 1, "I-biomedical": 2}
    else:
        raise ValueError(f"Task '{task}' not supported for cross-validation.")

    id2label = {v: k for k, v in label2id.items()}
    label_list = list(label2id.keys())
    model_id = (
        "meta-llama/Meta-Llama-3.1-8B"
        if llama_version == 3
        else "meta-llama/Llama-2-7b-hf"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    texts = ds_full["tokens"]
    labels = ds_full["ner_tags"]

    all_metrics = []

    for fold, (train_index, val_index) in enumerate(
        kf.split(texts, [l[0] if l else 0 for l in labels])
    ):
        print(f"Fold {fold+1}/{n_splits}")
        train_dataset = ds_full.select(train_index)
        val_dataset = ds_full.select(val_index)

        tokenized_train_ds = train_dataset.map(
            lambda examples: tokenize_and_align_labels_new(
                examples, tokenizer, label2id, max_length
            ),
            batched=True,
        )
        tokenized_val_ds = val_dataset.map(
            lambda examples: tokenize_and_align_labels_new(
                examples, tokenizer, label2id, max_length
            ),
            batched=True,
        )

        model_base = UnmaskingLlamaForTokenClassification.from_pretrained(
            lora_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
        ).bfloat16()
        model = PeftModel.from_pretrained(model_base, lora_path)
        model = model.merge_and_unload()
        model.resize_token_embeddings(len(tokenizer))

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=f"test_model_fold_{fold+1}",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="no",
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, label_list),
        )

        trainer.train()
        fold_metrics = trainer.evaluate()
        all_metrics.append(fold_metrics)

    avg_precision = np.mean([m["eval_precision"] for m in all_metrics])
    avg_recall = np.mean([m["eval_recall"] for m in all_metrics])
    avg_f1 = np.mean([m["eval_f1"] for m in all_metrics])
    avg_accuracy = np.mean([m["eval_accuracy"] for m in all_metrics])

    print(f"\nCross-Validation Results for {task}:")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")
    print(f"  Average F1 Score: {avg_f1:.4f}")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")

    return {
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": avg_f1,
        "average_accuracy": avg_accuracy,
        "fold_metrics": all_metrics,
    }


if __name__ == "__main__":
    task_name = "ncbi"  # You can change this to other supported tasks
    lora_path_val = "manumartinm/vaner"
    llama_version_val = 2
    epochs_val = 1
    batch_size_val = 4
    learning_rate_val = 1e-4
    max_length_val = 128
    lora_r_val = 12
    n_splits_val = 5

    results = cross_validate(
        task=task_name,
        lora_path=lora_path_val,
        llama_version=llama_version_val,
        epochs=epochs_val,
        batch_size=batch_size_val,
        learning_rate=learning_rate_val,
        max_length=max_length_val,
        lora_r=lora_r_val,
        n_splits=n_splits_val,
    )

    print("\nFinal Cross-Validation Summary:")
    print(f"Average Precision: {results['average_precision']:.4f}")
    print(f"Average Recall: {results['average_recall']:.4f}")
    print(f"Average F1 Score: {results['average_f1']:.4f}")
    print(f"Average Accuracy: {results['average_accuracy']:.4f}")
    print("\nFold-wise Metrics:")
    for i, metrics in enumerate(results["fold_metrics"]):
        print(
            f"  Fold {i+1}: Precision={metrics['eval_precision']:.4f}, Recall={metrics['eval_recall']:.4f}, F1={metrics['eval_f1']:.4f}, Accuracy={metrics['eval_accuracy']:.4f}"
        )

    with open("cross_validation_results.json", "w") as f:
        json.dump(results, f, indent=4)
