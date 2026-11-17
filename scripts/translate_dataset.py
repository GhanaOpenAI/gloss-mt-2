#!/usr/bin/env python3
"""High-throughput dataset translation CLI for Nkrane-MT models.

Optimized to saturate GPU capacity on ANY machine (single GPU, A100/H100/H200, T4/A10G, or CPU).
Translates large datasets from Parquet, JSONL, CSV, Text, or Hugging Face Hub in batches.

Usage:
    # Direct high-throughput batch translation of a Hugging Face dataset:
    python scripts/translate_dataset.py \
        --model ghanaopenai/nkrane-mt-nllb-600m-en2tw \
        --input databricks/databricks-dolly-15k \
        --split train \
        --src-column instruction \
        --tgt-column instruction_twi \
        --output dolly_twi.parquet \
        --batch-size 64

    # Translate a local JSONL or Parquet file:
    python scripts/translate_dataset.py \
        --model ghanaopenai/nkrane-mt-opus-en2tw \
        --input input_corpus.parquet \
        --src-column text \
        --tgt-column twi_text \
        --output translated.parquet \
        --batch-size 128
"""

import argparse
import os
import sys
import time
from typing import Dict, Iterator, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nkrane_mt.inference import NkraneMTTranslator


def iterate_input_rows(
    input_source: str,
    src_column: str,
    split: str = "train",
    limit: int = None,
) -> Iterator[Dict]:
    """Yields rows from HF dataset, parquet, jsonl, csv, or txt."""
    if os.path.isfile(input_source):
        ext = os.path.splitext(input_source)[1].lower()
        if ext == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(input_source)
            for idx, row in enumerate(table.to_pylist()):
                if limit and idx >= limit:
                    break
                yield row
        elif ext in (".jsonl", ".json"):
            import json

            count = 0
            with open(input_source, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if limit and count >= limit:
                        break
                    yield json.loads(line)
                    count += 1
        elif ext == ".csv":
            import csv

            with open(input_source, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if limit and idx >= limit:
                        break
                    yield row
        elif ext == ".txt":
            with open(input_source, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if limit and idx >= limit:
                        break
                    yield {src_column: line.strip()}
    else:
        # Load from Hugging Face Hub (streaming)
        from datasets import load_dataset

        ds = load_dataset(input_source, split=split, streaming=True)
        for idx, row in enumerate(ds):
            if limit and idx >= limit:
                break
            yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ghanaopenai/nkrane-mt-nllb-600m-en2tw", help="HF model ID or path")
    ap.add_argument("--input", required=True, help="Input file path (.parquet, .jsonl, .csv, .txt) or HF dataset ID")
    ap.add_argument("--output", required=True, help="Output path (.parquet or .jsonl)")
    ap.add_argument("--src-column", default="english", help="Column name containing English source text")
    ap.add_argument("--tgt-column", default="twi", help="Target column name for translated text")
    ap.add_argument("--split", default="train", help="Split to load if input is a Hugging Face dataset")
    ap.add_argument("--batch-size", default="auto", help="Batch size for GPU forward passes (int, or 'auto' to saturate GPU VRAM)")
    ap.add_argument("--max-length", type=int, default=512, help="Max sequence length (input and output, models trained up to 512)")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit for quick tests")
    ap.add_argument("--with-glossary", action="store_true", help="Enable offline dictionary entity substitution")
    ap.add_argument("--device", default=None, help="Device to use (cuda:0, cpu, mps)")
    ap.add_argument("--push", default=None, help="Optional HF dataset repo id to push output to")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading translator: {args.model} on {device}...", flush=True)

    translator = NkraneMTTranslator(
        model=args.model,
        use_default_dictionary=args.with_glossary,
        device=device,
    )

    skip_sub = not args.with_glossary
    auto_mark = args.with_glossary

    bs_arg = int(args.batch_size) if str(args.batch_size).isdigit() else args.batch_size

    print(f"Reading input from: {args.input} (column: {args.src_column})...", flush=True)
    print(f"Mode: {'Entity-Grounded Glossary' if args.with_glossary else 'Direct High-Speed (No Substitution)'}", flush=True)
    print(f"Batch size: {args.batch_size} | Device: {device}", flush=True)

    out_ext = os.path.splitext(args.output)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    # Initialize Parquet or JSONL streaming output
    parquet_writer = None
    jsonl_file = None
    if out_ext == ".jsonl":
        import json
        jsonl_file = open(args.output, "w", encoding="utf-8")

    total_translated = 0
    t0 = time.time()
    batch_rows = []

    def flush_batch(rows: List[Dict]):
        nonlocal total_translated, parquet_writer
        if not rows:
            return

        texts = [str(r.get(args.src_column, "") or "") for r in rows]
        translations = translator.translate_batch(
            texts=texts,
            batch_size=bs_arg,
            max_src_length=args.max_length,
            max_new_tokens=args.max_length,
            auto_mark=auto_mark,
            skip_substitution=skip_sub,
            return_details=False,
        )

        for r, tr in zip(rows, translations):
            r[args.tgt_column] = tr

        # Write out
        if out_ext == ".jsonl":
            for r in rows:
                jsonl_file.write(json.dumps(r, ensure_ascii=False) + "\n")
            jsonl_file.flush()
        else:
            # Default to parquet
            import pyarrow as pa
            import pyarrow.parquet as pq

            batch_table = pa.Table.from_pylist(rows)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(args.output, batch_table.schema)
            parquet_writer.write_table(batch_table)

        total_translated += len(rows)
        elapsed = time.time() - t0
        speed = total_translated / elapsed if elapsed > 0 else 0
        print(f"Translated {total_translated:,} rows ({speed:.1f} rows/sec)...", end="\r", flush=True)

    for row in iterate_input_rows(args.input, src_column=args.src_column, split=args.split, limit=args.limit):
        batch_rows.append(row)
        if len(batch_rows) >= args.batch_size:
            flush_batch(batch_rows)
            batch_rows = []

    if batch_rows:
        flush_batch(batch_rows)

    if parquet_writer is not None:
        parquet_writer.close()
    if jsonl_file is not None:
        jsonl_file.close()

    total_time = time.time() - t0
    avg_speed = total_translated / total_time if total_time > 0 else 0
    print(f"\nDone! Successfully translated {total_translated:,} rows to {args.output}")
    print(f"Total time: {total_time:.1f}s (Average throughput: {avg_speed:.1f} rows/sec)")

    if args.push:
        print(f"Uploading output dataset to Hugging Face Hub: {args.push}...", flush=True)
        from datasets import Dataset
        import pyarrow.parquet as pq

        table = pq.read_table(args.output)
        hf_ds = Dataset(table)
        hf_ds.push_to_hub(args.push)
        print(f"Published to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
