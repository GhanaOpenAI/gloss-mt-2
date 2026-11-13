"""Build the largest high-quality English -> Twi training dataset for GLOSS-MT-2.

Format:
  - input:   Pure, natural English sentence with NO brackets (no preprocessing):
             "Spain produces oil than any other country in the world."
  - target:  Asante Twi translation that uses the ENGLISH noun phrases in angle brackets:
             "<World> nyinaa no, <Spain> na wonya <oil> fi ngodua mu sen ɔman biara."
  - english: Clean unbracketed English sentence.
  - twi:     Clean human Asante Twi reference.

Strict Quality Criteria:
  - Both English and Twi sentences are 6 to 100 words in length.
  - No captions, study notes, or boilerplate headers.
  - Target contains at least one valid bracketed term: re.search(r"<[^<>\n]+>", target).
  - Deduplicated by normalized text.
"""

import argparse
import json
import os
import random
import re
import time
from typing import Dict, List, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset

PATTERN = re.compile(r"<([^<>\n]+)>")
BAD_PATTERNS = re.compile(
    r"(\[Picture|\[Mfonini|\[Box|\[Study Note|sankɔhwɛ|return visit|bible reading)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_chunk_dataset(dataset_name: str, align_cache_path: str) -> List[Dict]:
    print(f"Loading alignment cache from {align_cache_path}...")
    tw_to_al = {}
    with open(align_cache_path, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            res = it.get("result", {})
            btw = res.get("bracketed_twi")
            if btw and "mapping" in res:
                clean_tw = re.sub(r"\s+", " ", btw.replace("<", "").replace(">", "")).strip()
                tw_to_al[clean_tw] = res
    print(f"Loaded and indexed {len(tw_to_al):,} cache entries.")

    print(f"Loading document chunks from {dataset_name}...")
    ds = load_dataset(dataset_name)
    pairs = []
    seen = set()

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for r in ds[split]:
            en = r.get("english", "").strip()
            tw = r.get("twi", "").strip()
            if not en or not tw:
                continue

            clean_tw = re.sub(r"\s+", " ", tw).strip()
            al = tw_to_al.get(clean_tw)
            if not al or not al.get("mapping"):
                continue

            mapping = al["mapping"]
            valid_pairs = sorted(
                [(ts.strip(), ep.strip()) for ep, ts in mapping.items() if ts and isinstance(ts, str) and ts.strip()],
                key=lambda x: len(x[0]),
                reverse=True,
            )
            if not valid_pairs:
                continue

            # In Twi, replace each twi_span with <english_phrase>!
            table = {k.lower(): v for k, v in valid_pairs}
            alts_tw = [r"\s+".join(map(re.escape, k.split())) for k, v in valid_pairs]
            pat_tw = re.compile(r"(?<!\w)(?:" + "|".join(alts_tw) + r")(?!\w)", re.IGNORECASE)

            def sub_tw(m):
                matched = m.group(0)
                en_p = table.get(matched.lower(), matched)
                if matched[:1].isupper() and not en_p[:1].isupper():
                    en_p = en_p[:1].upper() + en_p[1:]
                return f"<{en_p}>"

            tw_target = pat_tw.sub(sub_tw, tw)

            # Must have at least one bracketed English term in the Twi target
            b_tgt = PATTERN.findall(tw_target)
            if not b_tgt:
                continue

            key = (normalize(en), normalize(tw))
            if key in seen:
                continue
            seen.add(key)

            pairs.append({
                "input": en,         # NO brackets in input!
                "target": tw_target,  # Twi translation with English noun phrases in <...>
                "english": en,
                "twi": tw,
                "source": "gloss-mt-doc-chunks",
            })

    print(f"Loaded {len(pairs):,} unique verified document chunks.")
    return pairs


def recover_afrigloss_mapping(src_raw: str, inp: str) -> Dict[str, str]:
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


def load_afrigloss_pairs(
    afrigloss_dir: str,
    seen_keys: Set[Tuple[str, str]],
    max_count: int = 600_000,
) -> List[Dict]:
    print(f"Loading and filtering sentence pairs from {afrigloss_dir}...")
    pairs = []
    total = 0

    for split in ["train", "validation", "test"]:
        fp = os.path.join(afrigloss_dir, f"{split}.jsonl")
        if not os.path.exists(fp):
            continue

        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)

                inp = row.get("input", "").strip()
                tgt = row.get("tgt", "").strip()
                src_raw = row.get("src_raw", "").strip()
                tgt_ref = row.get("tgt_ref", "").strip()

                en_words = src_raw.split()
                tw_words = tgt_ref.split()

                # 1. Length check
                if len(en_words) < 6 or len(tw_words) < 6 or len(en_words) > 100 or len(tw_words) > 100:
                    continue

                # 2. Skip captions and study notes
                if BAD_PATTERNS.search(src_raw) or BAD_PATTERNS.search(tgt_ref):
                    continue

                # 3. Not identical
                if src_raw.lower() == tgt_ref.lower():
                    continue

                mapping = recover_afrigloss_mapping(src_raw, inp)
                if not mapping:
                    continue

                # Twi target: in tgt, replace each <twi_term> with <en_phrase>
                twi_to_en = {v.lower(): k for k, v in mapping.items()}

                def sub_tw(m):
                    tw_t = m.group(1).strip()
                    en_p = twi_to_en.get(tw_t.lower(), tw_t)
                    if tw_t[:1].isupper() and not en_p[:1].isupper():
                        en_p = en_p[:1].upper() + en_p[1:]
                    return f"<{en_p}>"

                tw_target = PATTERN.sub(sub_tw, tgt)

                b_tgt = PATTERN.findall(tw_target)
                if not b_tgt:
                    continue

                key = (normalize(src_raw), normalize(tgt_ref))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                pairs.append({
                    "input": src_raw,     # NO brackets in input!
                    "target": tw_target,  # Twi translation with English noun phrases in <...>
                    "english": src_raw,
                    "twi": tgt_ref,
                    "source": "afrigloss-sentence-corpus",
                })

                if len(pairs) >= max_count:
                    break
        if len(pairs) >= max_count:
            break

    print(f"Filtered {len(pairs):,} unique high-quality sentence pairs from {total:,} rows.")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--chunk-dataset",
        default="ghanaopenai/gloss-mt-parallel-en-twi",
        help="Hugging Face repo id for document chunks",
    )
    ap.add_argument(
        "--align-cache",
        default="/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/build/gold_alignment_cache.jsonl",
        help="Path to gold_alignment_cache.jsonl",
    )
    ap.add_argument(
        "--afrigloss-dir",
        default="/mnt/volume_d2wey28/projects/afrigloss-mt/data/gloss_dataset_en_tw_full",
        help="Path to afrigloss full en_tw dataset directory",
    )
    ap.add_argument(
        "--out-dir",
        default="build/gloss-mt-en-tw-large",
        help="Local output directory",
    )
    ap.add_argument(
        "--push",
        default="ghanaopenai/gloss-mt-en-tw-large",
        help="Hugging Face dataset repository to push to",
    )
    ap.add_argument("--val-size", type=int, default=2500)
    ap.add_argument("--test-size", type=int, default=2500)
    args = ap.parse_args()

    t0 = time.time()

    # 1. Load document chunks (input: clean English, target: Twi with <en_noun>)
    chunk_items = load_chunk_dataset(args.chunk_dataset, args.align_cache)
    seen_keys = {(normalize(it["english"]), normalize(it["twi"])) for it in chunk_items}

    # 2. Load and filter massive sentence pairs
    afrigloss_items = load_afrigloss_pairs(args.afrigloss_dir, seen_keys)

    # 3. Combine both collections
    all_items = chunk_items + afrigloss_items
    print(f"\nTotal combined verified pairs: {len(all_items):,}")

    random.seed(42)
    random.shuffle(all_items)

    val_count = args.val_size
    test_count = args.test_size
    train_count = len(all_items) - val_count - test_count

    splits = {
        "train": all_items[:train_count],
        "validation": all_items[train_count : train_count + val_count],
        "test": all_items[train_count + val_count :],
    }

    print(f"Splits: train={len(splits['train']):,}, validation={len(splits['validation']):,}, test={len(splits['test']):,}")

    ds_dict = DatasetDict({
        k: Dataset.from_list(v) for k, v in splits.items()
    })

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving combined dataset to {args.out_dir}...")
    ds_dict.save_to_disk(args.out_dir)

    CARD = f"""---
language:
- en
- tw
- ak
license: cc-by-nc-4.0
task_categories:
- translation
tags:
- twi
- akan
- asante-twi
- machine-translation
- gloss-mt-2
- source-noun-preservation
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# GLOSS-MT-2 Large-Scale English → Asante Twi (Akan) Parallel Dataset

Massive, high-quality parallel dataset for training **GLOSS-MT-2** models.

### Paradigm
- **Input (`input`):** Raw English sentence with **NO brackets** (zero preprocessing):
  `Spain produces oil than any other country in the world.`
- **Target (`target`):** Fluent Asante Twi translation using the **English noun phrases** in angle brackets:
  `<World> nyinaa no, <Spain> na wonya <oil> fi ngodua mu sen ɔman biara.`
- **Clean Reference Source (`english`):** `Spain produces oil than any other country in the world.`
- **Clean Reference Target (`twi`):** `Wiase nyinaa no, Spain na wonya ngo fi ngodua mu sen ɔman biara.`

### Split Statistics
| Split | Samples | Source Composition |
|---|---|---|
| **train** | {len(splits['train']):,} | Document chunks + sentence pairs |
| **validation** | {len(splits['validation']):,} | Held-out evaluation pairs |
| **test** | {len(splits['test']):,} | Held-out benchmark test set |
| **Total** | **{len(all_items):,}** | |

### License
CC BY-NC 4.0
"""
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARD)

    if args.push:
        print(f"Pushing combined dataset to Hugging Face {args.push}...")
        ds_dict.push_to_hub(args.push)
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.push,
            repo_type="dataset",
        )
        print(f"Successfully pushed to https://huggingface.co/datasets/{args.push}!")

    print(f"Build complete in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
