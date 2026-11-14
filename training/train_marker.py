"""Train a lightweight token classification model to mark noun phrases in English inputs for GLOSS-MT-2."""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

LABEL_LIST = ["O", "B-PRESERVE", "I-PRESERVE"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def align_labels_with_tokens(raw_text: str, marked_text: str, tokenizer) -> Tuple[List[int], List[int]]:
    """
    Given unbracketed `raw_text` and bracketed `marked_text`,
    tokenizes raw_text and assigns B-PRESERVE, I-PRESERVE, or O to each subword token.
    """
    bracket_spans = []
    cursor = 0
    clean_marked = ""
    for m in re.finditer(r"<([^<>\n]+)>", marked_text):
        content = m.group(1)
        start_in_marked = m.start()
        # Find position in raw_text
        match_in_raw = raw_text.find(content, cursor)
        if match_in_raw != -1:
            bracket_spans.append((match_in_raw, match_in_raw + len(content)))
            cursor = match_in_raw + len(content)

    enc = tokenizer(raw_text, return_offsets_mapping=True, truncation=True, max_length=256)
    labels = []
    for start, end in enc["offset_mapping"]:
        if start == end:
            labels.append(-100)
            continue
        token_label = LABEL2ID["O"]
        for span_start, span_end in bracket_spans:
            if start >= span_start and end <= span_end:
                if start == span_start:
                    token_label = LABEL2ID["B-PRESERVE"]
                else:
                    token_label = LABEL2ID["I-PRESERVE"]
                break
        labels.append(token_label)

    return enc["input_ids"], labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    correct, total = 0, 0
    for preds, labs in zip(true_predictions, true_labels):
        for p, l in zip(preds, labs):
            total += 1
            if p == l:
                correct += 1

    return {
        "accuracy": round(correct / max(1, total), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilbert-base-uncased", help="Base transformer encoder (e.g. distilbert, deberta-v3-small)")
    ap.add_argument("--dataset", default="ghanaopenai/gloss-mt-2-en-tw-grand", help="Source dataset containing (english, input)")
    ap.add_argument("--out", default="checkpoints/noun-marker", help="Output directory")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-samples", type=int, default=100000, help="Max training samples to use")
    args = ap.parse_args()

    print(f"Loading dataset {args.dataset}...", flush=True)
    raw_ds = load_dataset(args.dataset)
    tok = AutoTokenizer.from_pretrained(args.model)

    def prepare_data(split_name, max_n=None):
        data = raw_ds[split_name]
        if max_n and len(data) > max_n:
            data = data.select(range(max_n))
        all_input_ids = []
        all_labels = []
        for r in data:
            raw_en = r.get("english") or r.get("en", "")
            marked_en = r.get("input", "")
            if not raw_en or not marked_en:
                continue
            input_ids, labels = align_labels_with_tokens(raw_en, marked_en, tok)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        return Dataset.from_dict({"input_ids": all_input_ids, "labels": all_labels})

    print("Preparing tokenized IOB datasets...", flush=True)
    train_ds = prepare_data("train", max_n=args.max_samples)
    val_ds = prepare_data("validation", max_n=3000)
    print(f"Train: {len(train_ds):,} samples | Validation: {len(val_ds):,} samples", flush=True)

    config = AutoConfig.from_pretrained(args.model, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID)
    model = AutoModelForTokenClassification.from_pretrained(args.model, config=config)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )

    print("Training Noun Phrase Marker model...", flush=True)
    trainer.train()

    final_dir = os.path.join(args.out, "final")
    print(f"Saving marker model to {final_dir}...", flush=True)
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print("Marker training complete!", flush=True)


if __name__ == "__main__":
    main()
