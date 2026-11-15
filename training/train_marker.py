"""Train a lightweight token classification model to mark noun phrases in English inputs for Nkrane-MT."""

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

LABEL_LIST = ["O", "B-PRESERVE", "I-PRESERVE"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def disk_guard(path: str, min_free_gb: float = 20.0):
    class DiskGuardCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if control.should_save or state.global_step % 50 == 0:
                free_gb = shutil.disk_usage(path).free / 1e9
                if free_gb < min_free_gb:
                    print(f"DiskGuard: {free_gb:.1f} GB free < {min_free_gb} GB. Stopping training.")
                    control.should_save = True
                    control.should_training_stop = True

    return DiskGuardCallback()


def align_labels_with_tokens(raw_text: str, marked_text: str, tokenizer) -> Tuple[List[int], List[int]]:
    """
    Given unbracketed `raw_text` and bracketed `marked_text`,
    tokenizes raw_text and assigns B-PRESERVE, I-PRESERVE, or O to each subword token.
    """
    bracket_spans = []
    cursor = 0
    for m in re.finditer(r"<([^<>\n]+)>", marked_text):
        content = m.group(1).strip()
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
    tp, fp, fn = 0, 0, 0
    for preds, labs in zip(true_predictions, true_labels):
        for p, l in zip(preds, labs):
            total += 1
            if p == l:
                correct += 1
            if p in ("B-PRESERVE", "I-PRESERVE"):
                if l == p:
                    tp += 1
                else:
                    fp += 1
            elif l in ("B-PRESERVE", "I-PRESERVE"):
                fn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    return {
        "accuracy": round(correct / max(1, total), 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilbert-base-uncased", help="Base transformer encoder (e.g. distilbert-base-uncased, microsoft/deberta-v3-small)")
    ap.add_argument("--dataset", default="data/en_tw_grand", help="Source dataset path (local dir or HF repo)")
    ap.add_argument("--out", default="checkpoints/noun-marker", help="Output directory")
    ap.add_argument("--epochs", type=float, default=10, help="Train to convergence")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-samples", type=int, default=200000, help="Max training samples")
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--early-stopping-patience", type=int, default=5)
    ap.add_argument("--push", default=None, help="Optional HF repo id to push to")
    args = ap.parse_args()

    print(f"Loading dataset from {args.dataset}...", flush=True)
    if os.path.isdir(args.dataset):
        raw_ds = load_from_disk(args.dataset)
    else:
        raw_ds = load_dataset(args.dataset)

    print(f"Loading tokenizer {args.model}...", flush=True)
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

    print(f"Preparing tokenized IOB datasets (up to {args.max_samples:,} train samples)...", flush=True)
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
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=500,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        tf32=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
        callbacks=[
            disk_guard(args.out, min_free_gb=20.0),
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        ],
    )

    print("Starting Noun Phrase Marker training loop...", flush=True)
    trainer.train()

    print("Evaluating best checkpoint...", flush=True)
    eval_res = trainer.evaluate()
    print("Final evaluation metrics:", eval_res, flush=True)

    final_dir = os.path.join(args.out, "final")
    print(f"Saving marker model and tokenizer to {final_dir}...", flush=True)
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    if args.push:
        print(f"Pushing marker model to Hugging Face {args.push}...", flush=True)
        trainer.push_to_hub(args.push)

    print("Marker training complete!", flush=True)


if __name__ == "__main__":
    main()
