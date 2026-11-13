"""Build and push the clean high-quality GLOSS-MT-2 dataset to Hugging Face.

Every sentence pair comes directly from contextual document alignment:
  - input:   Pure, natural English (NO brackets, no preprocessing):
             "The police continue to show that the two men will go to court and open the case."
  - target:  Asante Twi translation that uses the English noun phrases in angle brackets:
             "12. <Police> no kɔ so kyerɛ sɛ <men> mmienu no bɛkɔ <court> na wɔabue <case> no."
  - twi:     Clean human Asante Twi gold reference without brackets.
  - english: Clean human English gold reference without brackets.
"""

import json
import os
import re
import time
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

REPO_ID = "ghanaopenai/gloss-mt-2-en-tw"
CACHE_PATH = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/build/gold_alignment_cache.jsonl"
OUT_DIR = "/mnt/volume_d2wey28/projects/gloss-mt-2/data/en_tw_quality"


def format_clean_pair(en, tw, al):
    mapping = al.get("mapping", {})
    btw = al.get("bracketed_twi", "")
    if not mapping or not btw:
        return None, []

    twi_to_en = {}
    valid_pairs = []
    for ep, ts in mapping.items():
        if ts and isinstance(ts, str) and ts.strip():
            ep_s = ep.strip()
            ts_s = ts.strip()
            twi_to_en[ts_s.lower()] = ep_s
            valid_pairs.append((ep_s, ts_s))

    def sub_tw(m):
        tw_t = m.group(1).strip()
        en_p = twi_to_en.get(tw_t.lower(), tw_t)
        if tw_t[:1].isupper() and not en_p[:1].isupper():
            en_p = en_p[:1].upper() + en_p[1:]
        return f"<{en_p}>"

    tw_target = re.sub(r"<([^<>\n]+)>", sub_tw, btw)
    return tw_target, valid_pairs


def main():
    api = HfApi()
    print(f"Loading alignment cache from {CACHE_PATH}...")
    t0 = time.time()
    tw_to_al = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            res = it.get("result", {})
            btw = res.get("bracketed_twi")
            if btw and "mapping" in res:
                clean_tw = re.sub(r"\s+", " ", btw.replace("<", "").replace(">", "")).strip()
                tw_to_al[clean_tw] = res
    print(f"Loaded {len(tw_to_al):,} clean alignments in {time.time()-t0:.2f}s")

    print("Loading source dataset ghanaopenai/gloss-mt-parallel-en-twi...")
    ds = load_dataset("ghanaopenai/gloss-mt-parallel-en-twi")

    splits = {}
    for split in ["train", "validation", "test"]:
        print(f"Processing {split} split...")
        rows = []
        for r in ds[split]:
            en = r["english"].strip()
            tw = r["twi"].strip()
            clean_tw = re.sub(r"\s+", " ", tw).strip()
            al = tw_to_al.get(clean_tw)
            if not al:
                continue

            tw_target, terms = format_clean_pair(en, tw, al)
            if not terms or "<" not in tw_target:
                continue

            rows.append({
                "input": en,
                "target": tw_target,
                "english": en,
                "twi": tw,
            })

        print(f"  {split}: {len(rows):,} quality pairs")
        splits[split] = Dataset.from_list(rows)

    ds_dict = DatasetDict(splits)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Saving quality dataset locally to {OUT_DIR}...")
    ds_dict.save_to_disk(OUT_DIR)

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

# GLOSS-MT-2: High-Quality English → Asante Twi (Akan) Parallel Dataset

High-quality parallel dataset for **GLOSS-MT-2**, constructed strictly from sentence pairs where English noun phrases were extracted directly from the sentence and aligned to their exact contextual translations in the matching Twi reference.

### Paradigm
- **Input (`input`):** Raw, natural English sentence with **NO brackets** (zero preprocessing):
  `The police continue to show that the two men will go to court and open the case.`
- **Target (`target`):** Fluent Asante Twi translation using the **English noun phrases** in angle brackets:
  `12. <Police> no kɔ so kyerɛ sɛ <men> mmienu no bɛkɔ <court> na wɔabue <case> no.`
- **Clean Reference Target (`twi`):** `12. Police no kɔ so kyerɛ sɛ mmarima mmienu no bɛkɔ asɛnnibea na wɔabue asɛm no.`
- **Clean Reference Source (`english`):** `The police continue to show that the two men will go to court and open the case.`

### Split Statistics
| Split | Samples | Preserved Nouns |
|---|---|---|
| **train** | {len(splits['train']):,} | 100% contextually aligned sentence pairs |
| **validation** | {len(splits['validation']):,} | Held-out evaluation set |
| **test** | {len(splits['test']):,} | Held-out benchmark test set |
| **Total** | **{sum(len(v) for v in splits.values()):,}** | |

### Code Repository
[`GhanaOpenAI/gloss-mt-2`](https://github.com/GhanaOpenAI/gloss-mt-2)
"""

    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARD)

    print(f"Pushing quality dataset to Hugging Face {REPO_ID}...")
    ds_dict.push_to_hub(REPO_ID)
    api.upload_file(
        path_or_fileobj=CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Successfully pushed to https://huggingface.co/datasets/{REPO_ID}!")


if __name__ == "__main__":
    main()
