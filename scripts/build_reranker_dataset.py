"""Build Cross-Encoder Reranker dataset for Contextual Lexical Disambiguation in GLOSS-MT-2.

Given a Twi sentence containing an English noun phrase in angle brackets <noun>:
The model predicts which candidate Twi translation from the dictionary best fits into this Twi sentence.

Format:
  - text_a: Twi sentence containing the English noun phrase in <...>:
            "Ɛyɛ <history> a ɛfata sɛ yɛkae."
  - text_b: Candidate Asante Twi translation from dictionary:
            "abakɔsɛm"
  - label:  1.0 if correct translation in this sentence context, 0.0 otherwise.
"""

import argparse
import gzip
import json
import os
import random
import re
import time
from typing import Dict, List, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset


def load_multi_dictionary(dict_path: str) -> Dict[str, List[str]]:
    print(f"Loading dictionary from {dict_path}...", flush=True)
    table = {}
    with gzip.open(dict_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) > 1:
                key = " ".join(parts[0].lower().split())
                table[key] = [p.strip() for p in parts[1:] if p.strip()]
    print(f"Loaded {len(table):,} dictionary entries.", flush=True)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dataset", default="ghanaopenai/gloss-mt-2-en-tw", help="HF aligned dataset")
    ap.add_argument("--dict-path", default="nkrane_mt/data/twi_dictionary.tsv.gz", help="Multi-candidate dictionary")
    ap.add_argument("--out-dir", default="data/reranker_dataset", help="Output directory")
    ap.add_argument("--negatives-per-pos", type=int, default=3, help="Number of negative distractors per positive")
    ap.add_argument("--push", default=None, help="Optional HF dataset id to push to")
    args = ap.parse_args()

    t0 = time.time()
    multi_dict = load_multi_dictionary(args.dict_path)
    all_twi_vocab = [cand for cands in multi_dict.values() for cand in cands if len(cand) >= 2]
    random.seed(42)

    print(f"Loading aligned dataset from {args.source_dataset}...", flush=True)
    ds = load_dataset(args.source_dataset)

    pattern_brackets = re.compile(r"<([^<>\n]+)>")

    splits_data = {}
    for split in ds.keys():
        print(f"Building reranker samples for {split} split ({len(ds[split]):,} rows)...", flush=True)
        reranker_rows = []
        ranking_eval_rows = []

        for r in ds[split]:
            twi_context = r["target"].strip()  # Twi sentence with <English nouns>
            gold_twi = r["twi"].strip()        # Clean gold Twi reference

            en_nouns = pattern_brackets.findall(twi_context)
            if not en_nouns:
                continue

            for noun in set(en_nouns):
                noun_clean = noun.strip()
                norm_noun = " ".join(noun_clean.lower().split())

                candidates = multi_dict.get(norm_noun, [])
                if not candidates:
                    continue

                # Find which candidate was used in the gold Twi sentence
                gold_match = None
                for c in candidates:
                    if re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", gold_twi, re.IGNORECASE):
                        gold_match = c
                        break

                if not gold_match:
                    continue

                # Positive sample (label 1.0)
                reranker_rows.append({
                    "text_a": twi_context,
                    "text_b": gold_match,
                    "label": 1.0,
                    "noun_phrase": noun_clean,
                })

                # Hard negative samples from dictionary candidates (label 0.0)
                neg_cands = [c for c in candidates if c.lower() != gold_match.lower()]

                # If fewer than negatives_per_pos, sample lexical distractors from general Twi vocab
                while len(neg_cands) < args.negatives_per_pos:
                    rand_c = random.choice(all_twi_vocab)
                    if rand_c.lower() != gold_match.lower() and rand_c not in neg_cands:
                        neg_cands.append(rand_c)

                for neg in neg_cands[: args.negatives_per_pos]:
                    reranker_rows.append({
                        "text_a": twi_context,
                        "text_b": neg,
                        "label": 0.0,
                        "noun_phrase": noun_clean,
                    })

                ranking_eval_rows.append({
                    "context": twi_context,
                    "noun_phrase": noun_clean,
                    "positive": gold_match,
                    "candidates": [gold_match] + neg_cands[: args.negatives_per_pos],
                })

        print(f"  {split}: {len(reranker_rows):,} classification pairs | {len(ranking_eval_rows):,} disambiguation queries", flush=True)
        splits_data[split] = Dataset.from_list(reranker_rows)

    ds_dict = DatasetDict(splits_data)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving reranker dataset to {args.out_dir}...", flush=True)
    ds_dict.save_to_disk(args.out_dir)

    if args.push:
        print(f"Pushing to Hugging Face {args.push}...", flush=True)
        ds_dict.push_to_hub(args.push)

    print(f"Reranker dataset generated in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
