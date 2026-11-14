"""Train a Cross-Encoder Reranker for Contextual Lexical Disambiguation in GLOSS-MT-2."""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits.squeeze()))
    preds = (probs >= 0.5).astype(float)
    acc = np.mean(preds == labels)
    return {
        "accuracy": round(float(acc), 4),
    }


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


class BCETrainer(Trainer):
    """Custom Trainer implementing Binary Cross-Entropy with Logits loss."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Davlan/afro-xlmr-base", help="Pretrained encoder (AfroXLMR or XLM-RoBERTa)")
    ap.add_argument("--data", default="data/reranker_dataset", help="Path to reranker dataset")
    ap.add_argument("--out", default="checkpoints/reranker-afro-xlmr", help="Output directory")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument("--warmup-steps", type=int, default=500)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Cross-Encoder Reranker training on {args.model} ({device})...", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # Ensure < and > are preserved as special tokens for target noun marking
    for t in ["<", ">"]:
        if t not in tok.get_vocab():
            tok.add_tokens([t], special_tokens=False)

    print(f"Loading reranker dataset from {args.data}...", flush=True)
    if os.path.isdir(args.data):
        raw_ds = load_from_disk(args.data)
    else:
        raw_ds = load_dataset(args.data)

    print(f"Tokenizing train ({len(raw_ds['train']):,} pairs) and validation ({len(raw_ds['validation']):,} pairs)...", flush=True)

    def tokenize_fn(examples):
        enc = tok(
            examples["text_a"],
            examples["text_b"],
            max_length=args.max_length,
            truncation=True,
        )
        enc["labels"] = [float(lbl) for lbl in examples["label"]]
        return enc

    tokenized_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Tokenizing cross-encoder inputs",
    )

    config = AutoConfig.from_pretrained(args.model, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        tf32=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = BCETrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[
            disk_guard(args.out, min_free_gb=20.0),
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        ],
    )

    print("Starting cross-encoder training loop...", flush=True)
    trainer.train()

    print("Evaluating best model...", flush=True)
    results = trainer.evaluate()
    print("Final evaluation metrics:", results, flush=True)

    final_dir = os.path.join(args.out, "final")
    print(f"Saving final reranker model and tokenizer to {final_dir}...", flush=True)
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    with open(os.path.join(args.out, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Reranker training complete!", flush=True)


if __name__ == "__main__":
    main()
