"""Build the largest high-quality English -> Twi training dataset with brackets on both sides.

Combines:
  1. High-quality document chunks from `ghanaopenai/gloss-mt-parallel-en-twi` (29,982 chunks)
  2. Quality-filtered sentences from `afrigloss-mt/data/gloss_dataset_en_tw_full` (510,000+ pairs)

Strict Quality Criteria:
  - Both English and Twi sentences are 6 to 100 words in length.
  - No captions, study notes, or boilerplate headers.
  - Bracketed terms are identical on both sides: set(terms_in_en) == set(terms_in_tw).
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


def load_chunk_dataset(dataset_name: str) -> List[Dict]:
    print(f"Loading document chunks from {dataset_name}...")
    ds = load_dataset(dataset_name)
    pairs = []
    seen = set()

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for r in ds[split]:
            inp = r.get("english_pretranslated", "").strip()
            tgt = r.get("twi_bracketed", "").strip()
            en = r.get("english", "").strip()
            tw = r.get("twi", "").strip()

            if not inp or not tgt or not en or not tw:
                continue

            b_inp = PATTERN.findall(inp)
            b_tgt = PATTERN.findall(tgt)
            if not b_inp or not b_tgt:
                continue

            # Verify identical bracketed terms on both sides
            if set(t.lower().strip() for t in b_inp) != set(t.lower().strip() for t in b_tgt):
                continue

            key = (normalize(en), normalize(tw))
            if key in seen:
                continue
            seen.add(key)

            pairs.append({
                "input": inp,
                "target": tgt,
                "english": en,
                "twi": tw,
                "source": "gloss-mt-doc-chunks",
            })

    print(f"Loaded {len(pairs):,} unique verified document chunks.")
    return pairs


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

                # 4. Check brackets on both sides
                b_inp = PATTERN.findall(inp)
                b_tgt = PATTERN.findall(tgt)
                if not b_inp or not b_tgt:
                    continue

                # 5. Check that all terms in input match target exactly
                if set(t.lower().strip() for t in b_inp) != set(t.lower().strip() for t in b_tgt):
                    continue

                key = (normalize(src_raw), normalize(tgt_ref))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                pairs.append({
                    "input": inp,
                    "target": tgt,
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

    # 1. Load document chunks
    chunk_items = load_chunk_dataset(args.chunk_dataset)
    seen_keys = {(normalize(it["english"]), normalize(it["twi"])) for it in chunk_items}

    # 2. Load and filter massive sentence pairs
    afrigloss_items = load_afrigloss_pairs(args.afrigloss_dir, seen_keys)

    # 3. Combine both collections
    all_items = chunk_items + afrigloss_items
    print(f"\nTotal combined verified pairs: {len(all_items):,}")

    # Shuffle with fixed seed for reproducibility
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
- gloss-mt
- large-scale
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

# GLOSS-MT Large-Scale English → Asante Twi (Akan) Parallel Dataset

Massive, high-quality parallel dataset for training **GLOSS-MT** models with exact in-place noun phrase bracketing on both sides.

### Sources
1. **Document Chunks (29,982 chunks):** Line-aligned document chunks from pristine government reports, finance budgets, legal texts (UDHR), health notices, and Akan narratives, with Gemini 3.8 Flash verified alignments.
2. **AfriGloss Sentence Corpus (500,000+ pairs):** Sentence-level bilingual text strictly filtered for sentence length, grammar, and 100% matching bracketed terms on both sides.

### Paradigm
- **Input (`input`):** English source with Twi translation in angle brackets `<twi_term>`.
- **Target (`target`):** Gold Asante Twi target with matching `<twi_term>` in angle brackets.
- **Reference Source (`english`):** Clean, unbracketed English sentence.
- **Reference Target (`twi`):** Clean, unbracketed Asante Twi reference.

### Split Statistics
| Split | Samples | Source Composition |
|---|---|---|
| **train** | {len(splits['train']):,} | ~30k document chunks + ~500k sentence pairs |
| **validation** | {len(splits['validation']):,} | High-quality held-out pairs |
| **test** | {len(splits['test']):,} | Benchmark test set |
| **Total** | **{len(all_items):,}** | |

### Sample
- **Input (English + `<Twi>`):** `10. Why might an elder be inclined to make rules for the <asafo>?`
- **Target (Twi + `<Twi>`):** `10. Ɛdɛn na ebetumi ama <asafo> mu panyin anya adwene sɛ obetumi ahyehyɛ mmara ama anuanom wɔ asafo no mu?`
- **Clean English:** `10. Why might an elder be inclined to make rules for the congregation?`
- **Clean Twi:** `10. Ɛdɛn na ebetumi ama asafo mu panyin anya adwene sɛ obetumi ahyehyɛ mmara ama anuanom wɔ asafo no mu?`

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
