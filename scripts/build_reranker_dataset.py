"""Build Cross-Encoder Reranker dataset for Contextual Lexical Disambiguation.

Given an English sentence with a marked target noun phrase <noun>, predicts the correct
Asante Twi translation among candidate dictionary options.

Format:
  - text_a: English sentence with target noun enclosed in <...>:
            "The judge entered the <court> to begin the trial."
  - text_b: Candidate Asante Twi translation:
            "asɛnnibea"
  - label:  1.0 if correct translation in context, 0.0 otherwise.
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
    print(f"Loading dictionary from {dict_path}...")
    table = {}
    with gzip.open(dict_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) > 1:
                key = " ".join(parts[0].lower().split())
                table[key] = [p.strip() for p in parts[1:] if p.strip()]
    print(f"Loaded {len(table):,} dictionary entries.")
    return table


def mark_noun_in_sentence(sentence: str, noun: str) -> Optional[str]:
    """Enclose the target noun in <...> inside the sentence."""
    words = noun.split()
    if not words:
        return None
    pattern = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", re.IGNORECASE)
    # Mark only the first occurrence or match
    m = pattern.search(sentence)
    if m:
        start, end = m.span()
        return sentence[:start] + f"<{m.group(0)}>" + sentence[end:]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dataset", default="ghanaopenai/gloss-mt-2-en-tw", help="HF aligned dataset")
    ap.add_argument("--dict-path", default="gloss_mt_2/data/twi_dictionary.tsv.gz", help="Multi-candidate dictionary")
    ap.add_argument("--out-dir", default="data/reranker_dataset", help="Output directory")
    ap.add_argument("--negatives-per-pos", type=int, default=3, help="Number of negative distractors per positive")
    ap.add_argument("--push", default=None, help="Optional HF dataset id to push to")
    args = ap.parse_args()

    t0 = time.time()
    multi_dict = load_multi_dictionary(args.dict_path)
    all_twi_vocab = [cand for cands in multi_dict.values() for cand in cands if len(cand) >= 2]
    random.seed(42)

    print(f"Loading aligned dataset from {args.source_dataset}...")
    ds = load_dataset(args.source_dataset)

    pattern_brackets = re.compile(r"<([^<>\n]+)>")

    splits_data = {}
    for split in ds.keys():
        print(f"Building reranker samples for {split} split ({len(ds[split]):,} rows)...")
        reranker_rows = []
        ranking_eval_rows = []

        for r in ds[split]:
            en = r["english"].strip()
            tw = r["twi"].strip()
            tgt_bracketed = r["target"].strip()

            # The bracketed English nouns in the target sentence
            en_nouns = pattern_brackets.findall(tgt_bracketed)
            if not en_nouns:
                continue

            for noun in set(en_nouns):
                noun_clean = noun.strip()
                norm_noun = " ".join(noun_clean.lower().split())

                # Find the marked context
                marked_context = mark_noun_in_sentence(en, noun_clean)
                if not marked_context:
                    continue

                # Find the corresponding gold Twi word in tw:
                # We can check dictionary candidate options for norm_noun that appear in tw
                candidates = multi_dict.get(norm_noun, [])
                gold_tw = None
                for c in candidates:
                    # Check whole word match in tw
                    if re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", tw, re.IGNORECASE):
                        gold_tw = c
                        break

                if not gold_tw:
                    # If not in candidate list, check if single dictionary option or alignment
                    continue

                # Positive sample (label 1.0)
                reranker_rows.append({
                    "text_a": marked_context,
                    "text_b": gold_tw,
                    "label": 1.0,
                    "noun_phrase": noun_clean,
                })

                # Hard negative samples from dictionary (label 0.0)
                neg_cands = [c for c in candidates if c.lower() != gold_tw.lower()]

                # If we need more negatives, sample lexical distractors from Twi vocab
                while len(neg_cands) < args.negatives_per_pos:
                    rand_c = random.choice(all_twi_vocab)
                    if rand_c.lower() != gold_tw.lower() and rand_c not in neg_cands:
                        neg_cands.append(rand_c)

                for neg in neg_cands[: args.negatives_per_pos]:
                    reranker_rows.append({
                        "text_a": marked_context,
                        "text_b": neg,
                        "label": 0.0,
                        "noun_phrase": noun_clean,
                    })

                ranking_eval_rows.append({
                    "context": marked_context,
                    "noun_phrase": noun_clean,
                    "positive": gold_tw,
                    "candidates": [gold_tw] + neg_cands[: args.negatives_per_pos],
                })

        print(f"  {split}: {len(reranker_rows):,} classification pairs | {len(ranking_eval_rows):,} disambiguation queries")
        splits_data[split] = Dataset.from_list(reranker_rows)

    ds_dict = DatasetDict(splits_data)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving reranker dataset to {args.out_dir}...")
    ds_dict.save_to_disk(args.out_dir)

    if args.push:
        print(f"Pushing to Hugging Face {args.push}...")
        ds_dict.push_to_hub(args.push)

    print(f"Reranker dataset generated in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
