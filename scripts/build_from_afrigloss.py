"""Build a massive GLOSS-MT-2 dataset from afrigloss-mt parallel corpora (580k+ sentences).

Converts the afrigloss-mt dataset:
  en -> tw:
    source: src_raw (clean English)
    target: tgt with <twi_term> replaced by <en_phrase>
    gold:   tgt_ref (clean Twi)
  tw -> en:
    source: tgt_ref (clean Twi)
    target: input (English with <twi_term>)
    gold:   src_raw (clean English)
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

PATTERN = re.compile(r"<([^<>\n]+)>")


def recover_mapping(src_raw: str, inp: str) -> Dict[str, str]:
    """Find the English phrases that were replaced by each <twi_term> in inp."""
    parts = PATTERN.split(inp)
    if len(parts) <= 1:
        return {}
    regex_parts = [re.escape(parts[i]) for i in range(0, len(parts), 2)]
    full_regex = r"^" + r"(.+?)".join(regex_parts) + r"$"
    m = re.match(full_regex, src_raw, re.DOTALL)
    if m:
        twi_terms = [parts[i] for i in range(1, len(parts), 2)]
        en_phrases = [m.group(j + 1).strip() for j in range(len(twi_terms))]
        return dict(zip(en_phrases, twi_terms))
    return {}


def convert_row(
    row: Dict,
) -> Optional[Tuple[Dict, Dict]]:
    src_raw = row.get("src_raw", "").strip()
    tgt_ref = row.get("tgt_ref", "").strip()
    inp = row.get("input", "").strip()
    tgt = row.get("tgt", "").strip()

    if not src_raw or not tgt_ref or not inp or not tgt:
        return None

    mapping = recover_mapping(src_raw, inp)
    if not mapping:
        return None

    # 1. en -> tw target: in tgt, replace each <twi_term> with <en_phrase>
    twi_to_en = {v.lower(): k for k, v in mapping.items()}

    def sub_tw(m):
        twi_term = m.group(1).strip()
        en_phrase = twi_to_en.get(twi_term.lower(), twi_term)
        if twi_term[:1].isupper() and not en_phrase[:1].isupper():
            en_phrase = en_phrase[:1].upper() + en_phrase[1:]
        return f"<{en_phrase}>"

    tw_target = PATTERN.sub(sub_tw, tgt)

    # 2. tw -> en target: inp is already English with <twi_term>
    en_target = inp

    terms = [{"english": k, "twi": v} for k, v in mapping.items()]

    row_en2tw = {
        "source": src_raw,
        "target": tw_target,
        "gold": tgt_ref,
        "terms": terms,
    }
    row_tw2en = {
        "source": tgt_ref,
        "target": en_target,
        "gold": src_raw,
        "terms": terms,
    }
    return row_en2tw, row_tw2en


def process_file(file_path: str, max_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    en2tw_list = []
    tw2en_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            res = convert_row(row)
            if res is not None:
                r_en2tw, r_tw2en = res
                en2tw_list.append(r_en2tw)
                tw2en_list.append(r_tw2en)
            if max_samples and len(en2tw_list) >= max_samples:
                break
    return en2tw_list, tw2en_list


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--afrigloss-dir",
        default="/mnt/volume_d2wey28/projects/afrigloss-mt/data/gloss_dataset_en_tw_full",
        help="Path to afrigloss en_tw dataset directory",
    )
    ap.add_argument("--out-dir", default="build/preserved-dataset-large", help="Output directory")
    ap.add_argument("--max-train-samples", type=int, default=None, help="Optional limit for training samples")
    ap.add_argument("--push", default=None, help="Optional HF dataset repo prefix to push to")
    args = ap.parse_args()

    print(f"Loading afrigloss dataset from {args.afrigloss_dir}...")
    t0 = time.time()

    splits_en2tw = {}
    splits_tw2en = {}

    for split in ["train", "validation", "test"]:
        fp = os.path.join(args.afrigloss_dir, f"{split}.jsonl")
        if not os.path.exists(fp):
            print(f"Split {split} not found at {fp}, skipping")
            continue

        limit = args.max_train_samples if split == "train" else 5000
        print(f"Processing {split} split...")
        t_s = time.time()
        en2tw_items, tw2en_items = process_file(fp, max_samples=limit)
        print(f"  Converted {len(en2tw_items):,} samples in {time.time()-t_s:.1f}s")

        splits_en2tw[split] = Dataset.from_list(en2tw_items)
        splits_tw2en[split] = Dataset.from_list(tw2en_items)

    os.makedirs(args.out_dir, exist_ok=True)
    ds_en2tw = DatasetDict(splits_en2tw)
    ds_tw2en = DatasetDict(splits_tw2en)

    en2tw_path = os.path.join(args.out_dir, "en_to_tw")
    tw2en_path = os.path.join(args.out_dir, "tw_to_en")

    print(f"Saving large en_to_tw dataset to {en2tw_path}...")
    ds_en2tw.save_to_disk(en2tw_path)
    print(f"Saving large tw_to_en dataset to {tw2en_path}...")
    ds_tw2en.save_to_disk(tw2en_path)

    if args.push:
        print(f"Pushing to Hugging Face {args.push}-en-tw-large...")
        ds_en2tw.push_to_hub(f"{args.push}-en-tw-large")
        print(f"Pushing to Hugging Face {args.push}-tw-en-large...")
        ds_tw2en.push_to_hub(f"{args.push}-tw-en-large")

    print(f"All large datasets generated in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
