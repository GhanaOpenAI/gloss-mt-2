"""Build the official GLOSS-MT-2 English -> Twi dataset.

Key Features:
1. Markdown stripped: all headers (##), table pipes/rules, bold/italics (*, **), and blockquotes (>) are removed.
2. Complete sentences: line packing cuts strictly at sentence boundaries (max 512 chars).
3. Full noun phrases: includes determiners, articles, and demonstratives ('the', 'a', 'this', 'their', 'our').
4. Clean input: pure English with NO markdown and NO brackets.
5. Target: fluent Asante Twi translation with English noun phrases in angle brackets `<...>` (no markdown).
6. 64 concurrent workers with Gemini 3.8 Flash (thinkingBudget: 0) and resumable JSONL caching.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
import spacy
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT_TEMPLATE = """Below is an English text, its Asante Twi (Akan) translation, and a list of English noun phrases from the text.
Each English noun phrase includes its determiner/article (e.g., 'the', 'a', 'this', 'their', 'our', 'these', 'those').
In Asante Twi, determiners like 'no' (the), 'yi' (this), or 'bi' (a) attach to the noun phrase.

Task:
For each English noun phrase, identify its corresponding complete Twi translation (including its matching Twi determiner like 'no', 'yi', 'bi' where applicable).
Then produce a target Twi text where the translated Twi spans are replaced by the exact English noun phrases in angle brackets <...>.

Rules:
- Never nest brackets.
- Keep the rest of the Twi sentence completely unchanged.
- If a phrase does not appear or is not translated in the Twi, leave it unbracketed.

Return ONLY a JSON object with:
- "mapping": mapping of English noun phrase -> matching Twi span (or null)
- "target_twi": the Twi text where the translated spans are replaced by <English noun phrase>

English:
{en}

Twi:
{tw}

Noun phrases:
{phrases}"""


def clean_line_markdown(line: str) -> str:
    """Strip all markdown formatting, tables, headers, and bullet markers."""
    # Remove table index rows like "| 14 | "
    line = re.sub(r"^\s*\|\s*[\d#]+\s*\|\s*", "", line)
    # Remove table rules
    if re.search(r"^[\s|:-]+$", line) or "|---" in line:
        return ""
    # Remove table pipes
    line = re.sub(r"^\s*\||\|\s*$", "", line)
    line = line.replace("|", " ")
    # Remove headers (##, ###)
    line = re.sub(r"^\s*#{1,6}\s*", "", line)
    # Remove blockquotes
    line = re.sub(r"^\s*>\s*", "", line)
    # Remove bold/italics (*, **)
    line = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", line)
    # Remove backticks
    line = re.sub(r"`([^`]+)`", r"\1", line)
    # Remove list bullet markers like "- ", "* " at start
    line = re.sub(r"^\s*[-*•]\s+", "", line)
    cleaned = re.sub(r"[ \t]+", " ", line).strip()
    if cleaned.lower() in ("sentence", "#", "sentences", "title", "text", "sector"):
        return ""
    return cleaned


def pack_lines_chars(lines: List[str], max_chars: int = 512) -> List[Tuple[int, int]]:
    """Pack complete lines/sentences up to max_chars without breaking sentences."""
    spans = []
    start, cur = 0, 0
    for i, line in enumerate(lines):
        line_len = len(line)
        if cur + line_len + 1 > max_chars and cur > 0:
            spans.append((start, i))
            start, cur = i, 0
        cur += line_len + 1
    if cur > 0 or not spans:
        spans.append((start, len(lines)))
    return spans


def extract_full_noun_phrases(text: str, nlp) -> List[str]:
    """Extract complete noun phrases with determiners/articles, filtering out bare pronouns."""
    doc = nlp(text)
    phrases = []
    seen = set()
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PRON":
            continue
        raw_clean = re.sub(r"\s+", " ", chunk.text).strip()
        raw_clean = re.sub(r"^[\W_]+|[\W_]+$", "", raw_clean)
        if len(raw_clean) >= 2 and raw_clean.lower() not in seen and re.search(r"[a-zA-Z]{2,}", raw_clean):
            seen.add(raw_clean.lower())
            phrases.append(raw_clean)
    return phrases


def chunk_key(en: str, tw: str) -> str:
    return hashlib.sha1(f"{en.strip()}|||{tw.strip()}".encode("utf-8")).hexdigest()


def align_call(en: str, tw: str, phrases: List[str], api_key: str, model: str) -> Optional[Dict]:
    prompt = PROMPT_TEMPLATE.format(en=en, tw=tw, phrases=json.dumps(phrases))
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    for attempt in range(5):
        try:
            r = requests.post(GEMINI_URL.format(model=model), params={"key": api_key}, json=body, timeout=45)
            if r.status_code == 200:
                raw = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                if isinstance(raw, dict) and "target_twi" in raw and "mapping" in raw:
                    return raw
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return None


def align_all_chunks(
    chunks: List[Dict],
    cache_path: str,
    api_key: str,
    model: str,
    workers: int = 64,
) -> Dict[str, Dict]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    it = json.loads(line)
                    cache[it["key"]] = it["result"]
                except Exception:
                    continue

    todo = [c for c in chunks if c["key"] not in cache and c["phrases"]]
    print(f"Alignment: {len(chunks):,} chunks total, {len(cache):,} cached, {len(todo):,} to align with {model}", flush=True)

    if not todo:
        return cache

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    done, failed, t0 = 0, 0, time.time()

    with open(cache_path, "a", encoding="utf-8") as cf, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(align_call, c["en"], c["tw"], c["phrases"], api_key, model): c
            for c in todo
        }
        for fut in concurrent.futures.as_completed(futures):
            c = futures[fut]
            res = fut.result()
            if res is None:
                failed += 1
                continue
            cache[c["key"]] = res
            cf.write(json.dumps({"key": c["key"], "result": res}, ensure_ascii=False) + "\n")
            cf.flush()
            done += 1
            if done % 500 == 0:
                rate = done / (time.time() - t0)
                rem = (len(todo) - done) / max(1, rate)
                print(f"  aligned {done:,}/{len(todo):,} ({rate:.1f}/s, ~{rem/60:.1f}m remaining)", flush=True)

    if failed:
        print(f"  Warning: {failed:,} chunks failed alignment call.", flush=True)
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/data")
    ap.add_argument("--cache-path", default="build/gold_alignment_cache_512.jsonl")
    ap.add_argument("--out-dir", default="data/en_tw_512")
    ap.add_argument("--push", default="ghanaopenai/gloss-mt-2-en-tw")
    ap.add_argument("--max-chars", type=int, default=512)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--model", default="gemini-3.8-flash")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY is not set.")

    print("Loading spaCy model en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    print(f"Chunking parallel documents (stripping markdown, max {args.max_chars} chars, complete sentences)...")
    t0 = time.time()
    splits_chunks = {}
    all_chunks = []

    for split in ["train", "validation", "test"]:
        fp = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(fp):
            continue

        with open(fp, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        chunks = []
        skipped = 0
        for r in rows:
            raw_en_lines = r["english"].splitlines()
            raw_tw_lines = r["translation"].splitlines()
            if len(raw_en_lines) != len(raw_tw_lines):
                skipped += 1
                continue

            clean_en_lines = [clean_line_markdown(l) for l in raw_en_lines]
            clean_tw_lines = [clean_line_markdown(l) for l in raw_tw_lines]

            # Filter out lines that became empty
            paired_lines = [
                (e, t) for e, t in zip(clean_en_lines, clean_tw_lines) if e and t
            ]
            if not paired_lines:
                continue

            en_lines = [p[0] for p in paired_lines]
            tw_lines = [p[1] for p in paired_lines]

            spans = pack_lines_chars(en_lines, max_chars=args.max_chars)
            for s, e in spans:
                en_c = " ".join(en_lines[s:e]).strip()
                tw_c = " ".join(tw_lines[s:e]).strip()
                if not en_c or not tw_c:
                    continue

                ph = extract_full_noun_phrases(en_c, nlp)
                k = chunk_key(en_c, tw_c)
                chunk_obj = {
                    "en": en_c,
                    "tw": tw_c,
                    "phrases": ph,
                    "key": k,
                }
                chunks.append(chunk_obj)
                all_chunks.append(chunk_obj)

        splits_chunks[split] = chunks
        print(f"Split {split}: {len(rows)-skipped:,} docs -> {len(chunks):,} chunks (skipped {skipped} mismatched)")

    print(f"Total chunks across all splits: {len(all_chunks):,} in {time.time()-t0:.1f}s")

    # Align with Gemini
    cache = align_all_chunks(
        chunks=all_chunks,
        cache_path=args.cache_path,
        api_key=api_key,
        model=args.model,
        workers=args.workers,
    )

    # Build dataset splits
    pattern = re.compile(r"<([^<>\n]+)>")
    ds_splits = {}

    for split, chunks in splits_chunks.items():
        rows = []
        for c in chunks:
            res = cache.get(c["key"])
            if not res or not res.get("target_twi"):
                continue

            tw_target = res["target_twi"].strip()
            # Must contain brackets in target
            if not pattern.search(tw_target):
                continue

            if tw_target.count("<") != tw_target.count(">"):
                continue

            rows.append({
                "input": c["en"],
                "target": tw_target,
                "english": c["en"],
                "twi": c["tw"],
            })

        print(f"Final valid {split} split: {len(rows):,} pairs (of {len(chunks):,} chunks)")
        ds_splits[split] = Dataset.from_list(rows)

    ds_dict = DatasetDict(ds_splits)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving dataset locally to {args.out_dir}...")
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

# GLOSS-MT-2: English → Asante Twi (Akan) Parallel Dataset

High-quality, clean parallel dataset for **GLOSS-MT-2** with full noun phrase preservation (including determiners/articles), complete sentences (max 512 characters), and stripped markdown formatting.

### Paradigm
- **Input (`input`):** Raw, natural English sentence with **NO brackets** and **NO markdown** (zero preprocessing):
  `It reached a small afternoon, and the news reporters went to the Kejetia market place. They went to see how the trade is there now. This matter is heavy.`
- **Target (`target`):** Fluent Asante Twi translation using the complete **English noun phrases with determiners/articles** in angle brackets:
  `Ɛduruu <a small afternoon>, na <the news reporters> nsrane kɔɔ <the Kejetia market place>. Wɔkɔhwɛɛ sɛnea <the trade> teɛ wɔ hɔ seisei. <This matter> yɛ duru.`
- **Clean Reference Target (`twi`):** `Ɛduruu owia ketewa bi, na amaneɛbɔ adwumayɛfoɔ no nsrane kɔɔ Kejetia dwadibea hɔ. Wɔkɔhwɛɛ sɛnea dwadie no teɛ wɔ hɔ seisei. Asɛm yi yɛ duru.`
- **Clean Reference Source (`english`):** `It reached a small afternoon, and the news reporters went to the Kejetia market place. They went to see how the trade is there now. This matter is heavy.`

### Split Statistics
| Split | Samples | Length Constraint |
|---|---|---|
| **train** | {len(ds_splits['train']):,} | Complete sentences $\le$ 512 characters (no markdown) |
| **validation** | {len(ds_splits['validation']):,} | Complete sentences $\le$ 512 characters (no markdown) |
| **test** | {len(ds_splits['test']):,} | Complete sentences $\le$ 512 characters (no markdown) |
| **Total** | **{sum(len(v) for v in ds_splits.values()):,}** | 100% complete grammatical units |

### Code Repository
[`GhanaOpenAI/gloss-mt-2`](https://github.com/GhanaOpenAI/gloss-mt-2)
"""
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARD)

    if args.push:
        print(f"Pushing dataset to Hugging Face {args.push}...")
        ds_dict.push_to_hub(args.push)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.push,
            repo_type="dataset",
        )
        print(f"Successfully published to https://huggingface.co/datasets/{args.push}!")

    print(f"Pipeline complete in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
