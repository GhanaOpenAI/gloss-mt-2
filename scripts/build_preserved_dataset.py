"""Build parallel datasets for GLOSS-MT-2.

The model receives clean, un-preprocessed source text (no brackets or markup).
The model is trained to generate the translation with source noun phrases left
untranslated and enclosed in angle brackets `<...>` for post-translation glossary substitution.
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


def format_targets(
    en: str,
    tw: str,
    mapping: Dict[str, Optional[str]],
    bracketed_twi: str,
) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    1. en_to_tw_target: Twi text where each <twi_span> is replaced by its English noun phrase in <en_phrase>.
    2. tw_to_en_target: English text where each english_phrase is replaced by <twi_span>.
    """
    twi_to_en = {}
    valid_pairs = []
    for ep, ts in mapping.items():
        if ts and isinstance(ts, str) and ts.strip():
            ep_s = ep.strip()
            ts_s = ts.strip()
            twi_to_en[ts_s.lower()] = ep_s
            valid_pairs.append((ep_s, ts_s))

    # 1. en -> tw: replace <twi_span> with <en_phrase>
    def sub_tw(m):
        span = m.group(1).strip()
        en_p = twi_to_en.get(span.lower(), span)
        if span[:1].isupper() and not en_p[:1].isupper():
            en_p = en_p[:1].upper() + en_p[1:]
        return f"<{en_p}>"

    tw_target = re.sub(r"<([^<>\n]+)>", sub_tw, bracketed_twi)

    # 2. tw -> en: replace english_phrase with <twi_span>
    valid_pairs.sort(key=lambda x: len(x[0]), reverse=True)
    table = {k.lower(): v for k, v in valid_pairs}
    if table:
        alts = [r"\s+".join(map(re.escape, k.split())) for k in sorted(table.keys(), key=len, reverse=True)]
        pat = re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE)

        def sub_en(m):
            matched = m.group(0)
            k = " ".join(matched.lower().split())
            tw_span = table.get(k, matched)
            if matched[:1].isupper() and not tw_span[:1].isupper():
                tw_span = tw_span[:1].upper() + tw_span[1:]
            return f"<{tw_span}>"

        en_target = pat.sub(sub_en, en)
    else:
        en_target = en

    terms = [{"english": p, "twi": s} for p, s in valid_pairs]
    return tw_target, en_target, terms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-dataset",
        default="ghanaopenai/gloss-mt-parallel-en-twi",
        help="HF source dataset with clean english and twi text",
    )
    ap.add_argument("--align-cache", required=True, help="Path to gold_alignment_cache.jsonl")
    ap.add_argument("--out-dir", default="build/preserved-dataset", help="Output directory")
    ap.add_argument("--push", default=None, help="Hugging Face repo prefix to push to (optional)")
    args = ap.parse_args()

    print(f"Loading alignment cache from {args.align_cache}...")
    t0 = time.time()
    tw_to_al = {}
    with open(args.align_cache, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            res = it.get("result", {})
            btw = res.get("bracketed_twi")
            if btw and "mapping" in res:
                clean_tw = re.sub(r"\s+", " ", btw.replace("<", "").replace(">", "")).strip()
                tw_to_al[clean_tw] = res
    print(f"Loaded and indexed {len(tw_to_al):,} cache entries in {time.time()-t0:.2f}s")

    print(f"Loading source dataset from {args.source_dataset}...")
    t1 = time.time()
    ds = load_dataset(args.source_dataset)
    print(f"Loaded dataset in {time.time()-t1:.2f}s")

    splits_en2tw = {}
    splits_tw2en = {}

    for split in ds.keys():
        print(f"Processing {split} split ({len(ds[split]):,} rows)...")
        en2tw_rows = []
        tw2en_rows = []
        matched_count = 0

        for i, r in enumerate(ds[split]):
            en = r["english"]
            tw = r["twi"]
            clean_tw = re.sub(r"\s+", " ", tw).strip()
            al = tw_to_al.get(clean_tw)

            if al and al.get("bracketed_twi") and al.get("mapping"):
                tw_target, en_target, terms = format_targets(en, tw, al["mapping"], al["bracketed_twi"])
                matched_count += 1
            else:
                tw_target = tw
                en_target = en
                terms = []

            row_id = f"{split}_{i}"

            en2tw_rows.append({
                "id": row_id,
                "source": en,
                "target": tw_target,
                "gold": tw,
                "terms": terms,
            })

            tw2en_rows.append({
                "id": row_id,
                "source": tw,
                "target": en_target,
                "gold": en,
                "terms": terms,
            })

        print(f"  {split}: {matched_count:,} of {len(ds[split]):,} rows ({matched_count/len(ds[split]):.1%}) aligned with preserved nouns")
        splits_en2tw[split] = Dataset.from_list(en2tw_rows)
        splits_tw2en[split] = Dataset.from_list(tw2en_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    ds_en2tw = DatasetDict(splits_en2tw)
    ds_tw2en = DatasetDict(splits_tw2en)

    en2tw_path = os.path.join(args.out_dir, "en_to_tw")
    tw2en_path = os.path.join(args.out_dir, "tw_to_en")

    print(f"Saving en_to_tw dataset to {en2tw_path}...")
    ds_en2tw.save_to_disk(en2tw_path)
    print(f"Saving tw_to_en dataset to {tw2en_path}...")
    ds_tw2en.save_to_disk(tw2en_path)

    if args.push:
        print(f"Pushing to Hugging Face {args.push}-en-tw...")
        ds_en2tw.push_to_hub(f"{args.push}-en-tw")
        print(f"Pushing to Hugging Face {args.push}-tw-en...")
        ds_tw2en.push_to_hub(f"{args.push}-tw-en")

    print("All datasets generated successfully!")


if __name__ == "__main__":
    main()
