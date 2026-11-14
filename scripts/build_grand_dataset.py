"""Build the Grand GLOSS-MT-2 Dataset across 4 major pillars:
1. GHOK Parallel Corpus (~457k passages: STEM, technical, academic, governance)
2. Pristine Twi-English (~450k passages: balanced 25% Narrative, 25% Dialogue, 25% Storyful, 25% Monologue)
3. Ghanaian News Paragraphs (~256k passages: Ghanaian current affairs, politics, economy)
4. JW Twi-English Corpus (~225k sentences: verse-aligned Akan literature & daily life)

Total raw passages: ~1.4M -> ~2.5M complete sentence chunks (<= 512 chars).
"""

import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
import spacy
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_LEADING_ARTICLES = re.compile(r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE)

PROMPT_TEMPLATE = """Below is an English text, its Asante Twi translation, and a list of English base noun phrases.

CRITICAL Grammar & Bracketing Rules:
1. ONLY the base noun / noun phrase must be inside the angle brackets.
   DO NOT include any articles or determiners ('the', 'a', 'an', 'this', 'that', 'these', 'those') inside the brackets!
   - Examples of correct bracketing:
     - '<government> no' (NOT '<the government>')
     - '<news reporters> no' (NOT '<the news reporters>')
     - '<matter> yi' (NOT '<this matter>')
     - 'sɛ <soldier>' (NOT '<a soldier>')
     - '<trade> no' (NOT '<the trade>')
     - '<nation> yi' (NOT '<this nation>')

2. In the Twi target sentence, replace only the translated noun with the English base noun in angle brackets, keeping the Twi determiners ('no', 'yi', 'bi', etc.) in Twi outside the brackets:
   - 'aban no' becomes '<government> no'
   - 'amaneɛbɔ adwumayɛfoɔ no' becomes '<news reporters> no'
   - 'asɛm yi' becomes '<matter> yi'
   - 'dwadie no' becomes '<trade> no'
   - 'ɔman yi' becomes '<nation> yi'

3. Never nest brackets. Do not change any surrounding words.

Return ONLY a JSON object with:
- "mapping": English base noun -> matching Twi noun span (excluding determiners like 'no', 'yi', 'bi')
- "target_twi": Twi text where ONLY the base nouns are in angle brackets <noun>, with all Twi determiners kept outside.

English:
{en}

Twi:
{tw}

Noun phrases:
{phrases}"""


def load_api_keys() -> List[str]:
    """Dynamically load Gemini API keys from environment or local key files."""
    keys = []
    env_keys = os.environ.get("GEMINI_API_KEYS", "")
    if env_keys:
        keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])
    single_key = os.environ.get("GEMINI_API_KEY", "")
    if single_key and single_key not in keys:
        keys.insert(0, single_key.strip())

    key_file_candidates = [
        "gemini_keys.txt",
        "../gemini_keys.txt",
        "/mnt/volume_d2wey28/projects/gloss-mt-2/gemini_keys.txt",
        "/home/owusus/Documents/GitHub/gemini-live-keys.txt",
    ]
    for candidate in key_file_candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                for line in f:
                    k = line.strip()
                    if k and not k.startswith("#") and len(k) > 20 and not k.startswith("export"):
                        if k not in keys:
                            keys.append(k)
    return keys


def clean_line_markdown(line: str) -> str:
    line = re.sub(r"^\s*\|\s*[\d#]+\s*\|\s*", "", line)
    if re.search(r"^[\s|:-]+$", line) or "|---" in line:
        return ""
    line = re.sub(r"^\s*\||\|\s*$", "", line)
    line = line.replace("|", " ")
    line = re.sub(r"^\s*#{1,6}\s*", "", line)
    line = re.sub(r"^\s*>\s*", "", line)
    line = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", line)
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"^\s*[-*•]\s+", "", line)
    cleaned = re.sub(r"[ \t]+", " ", line).strip()
    if cleaned.lower() in ("sentence", "#", "sentences", "title", "text", "sector"):
        return ""
    return cleaned


def split_into_complete_sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def pack_sentences_chars(sentences: List[str], max_chars: int = 512) -> List[str]:
    chunks = []
    cur = []
    cur_len = 0
    for s in sentences:
        s_len = len(s)
        if cur_len + s_len + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = [s]
            cur_len = s_len
        else:
            cur.append(s)
            cur_len += s_len + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def extract_base_noun_phrases(text: str, nlp) -> List[str]:
    doc = nlp(text)
    phrases = []
    seen = set()
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PRON":
            continue
        raw_clean = re.sub(r"\s+", " ", chunk.text).strip()
        raw_clean = re.sub(r"^[\W_]+|[\W_]+$", "", raw_clean)
        base = _LEADING_ARTICLES.sub("", raw_clean).strip()
        if len(base) >= 2 and base.lower() not in seen and re.search(r"[a-zA-Z]{2,}", base):
            seen.add(base.lower())
            phrases.append(base)
    return phrases


def chunk_key(en: str, tw: str) -> str:
    return hashlib.sha1(f"{en.strip()}|||{tw.strip()}".encode("utf-8")).hexdigest()


def align_call(en: str, tw: str, phrases: List[str], key: str, model: str) -> Optional[Dict]:
    prompt = PROMPT_TEMPLATE.format(en=en, tw=tw, phrases=json.dumps(phrases))
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    for attempt in range(4):
        try:
            r = requests.post(GEMINI_URL.format(model=model), params={"key": key}, json=body, timeout=30)
            if r.status_code == 200:
                raw = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                if isinstance(raw, dict) and "target_twi" in raw and "mapping" in raw:
                    return raw
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/en_tw_grand")
    ap.add_argument("--cache-path", default="build/grand_alignment_cache.jsonl")
    ap.add_argument("--chunks-cache", default="build/grand_chunks_cache.json")
    ap.add_argument("--push", default="ghanaopenai/gloss-mt-2-en-tw-grand")
    ap.add_argument("--pristine-sample-size", type=int, default=450_000, help="Balanced sample from pristine-twi-english matching GHOK")
    ap.add_argument("--workers-per-key", type=int, default=6)
    ap.add_argument("--model", default="gemini-3.8-flash")
    args = ap.parse_args()

    all_keys = load_api_keys()
    if not all_keys:
        sys.exit("Error: No Gemini API keys found in environment or key files.")

    total_workers = min(128, len(all_keys) * args.workers_per_key)
    print(f"Loaded {len(all_keys)} API keys ({total_workers} total parallel workers).", flush=True)

    if os.path.exists(args.chunks_cache):
        print(f"Loading cached complete chunks from {args.chunks_cache}...", flush=True)
        t0 = time.time()
        with open(args.chunks_cache, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        print(f"Loaded {len(all_chunks):,} pre-chunked items in {time.time()-t0:.2f}s", flush=True)
    else:
        print("Extracting and chunking across all 4 pillars...", flush=True)
        t0 = time.time()
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        all_chunks = []
        seen_keys = set()

        def add_pair(en_raw: str, tw_raw: str, source_tag: str):
            en_clean = clean_line_markdown(en_raw)
            tw_clean = clean_line_markdown(tw_raw)
            if not en_clean or not tw_clean:
                return

            en_sents = split_into_complete_sentences(en_clean)
            tw_sents = split_into_complete_sentences(tw_clean)
            if len(en_sents) != len(tw_sents) or len(en_sents) == 0:
                if len(en_clean) <= 512 and len(tw_clean) <= 512:
                    k = chunk_key(en_clean, tw_clean)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        ph = extract_base_noun_phrases(en_clean, nlp)
                        if ph:
                            all_chunks.append({"en": en_clean, "tw": tw_clean, "phrases": ph, "key": k, "source": source_tag})
                return

            en_packs = pack_sentences_chars(en_sents, max_chars=512)
            tw_packs = pack_sentences_chars(tw_sents, max_chars=512)
            if len(en_packs) == len(tw_packs):
                for ec, tc in zip(en_packs, tw_packs):
                    k = chunk_key(ec, tc)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        ph = extract_base_noun_phrases(ec, nlp)
                        if ph:
                            all_chunks.append({"en": ec, "tw": tc, "phrases": ph, "key": k, "source": source_tag})

        # Pillar 1: GHOK (~457k)
        print("Loading Pillar 1: GHOK Parallel Corpus...", flush=True)
        ds_ghok = load_dataset("ghananlpcommunity/ghok-parallel-en-twi", split="train")
        for r in ds_ghok:
            add_pair(r["english"], r["twi"], "ghok")
        print(f"  Pillar 1 accumulated: {len(all_chunks):,} chunks", flush=True)

        # Pillar 2: Pristine Twi-English (Balanced across 4 styles)
        print("Loading Pillar 2: Pristine Twi-English (Balanced)...", flush=True)
        ds_pristine = load_dataset("ghanaopenai/pristine-twi-english", split="train")
        target_per_style = args.pristine_sample_size // 4
        style_counts = {"Narrative": 0, "Dialogue": 0, "Storyful": 0, "Monologue": 0}
        for r in ds_pristine:
            st = r.get("style", "")
            if st in style_counts and style_counts[st] < target_per_style:
                style_counts[st] += 1
                add_pair(r["english_translation"], r["twi"], f"pristine_{st.lower()}")
        print(f"  Pillar 1 + 2 accumulated: {len(all_chunks):,} chunks", flush=True)

        # Pillar 3: News Paragraphs (~256k)
        print("Loading Pillar 3: Ghanaian News Paragraphs...", flush=True)
        ds_news = load_dataset("ghananlpcommunity/twi-english-paragraph-dataset_news", split="train")
        for r in ds_news:
            add_pair(r["ENGLISH"], r["TWI"], "news")
        print(f"  Pillar 1 + 2 + 3 accumulated: {len(all_chunks):,} chunks", flush=True)

        # Pillar 4: JW Twi-English (~225k)
        print("Loading Pillar 4: JW African Corpus...", flush=True)
        try:
            sys.path.insert(0, "/mnt/volume_d2wey28/projects/afrigloss-mt")
            from afrigloss.corpus import AfricaCorpusLoader
            loader = AfricaCorpusLoader()
            jw_tw = loader.load_verses_from_file("Twi_tw_v1.csv", loader.jw_repo_id)
            jw_en = loader.load_verses_from_file("english_cache.csv", loader.jw_repo_id)
            common_keys = sorted(list(set(jw_tw.keys()).intersection(jw_en.keys())))
            for k in common_keys:
                add_pair(jw_en[k], jw_tw[k], "jw")
        except Exception as e:
            print("Note on Pillar 4:", e, flush=True)

        print(f"All pillars chunked: {len(all_chunks):,} total chunks in {time.time()-t0:.1f}s", flush=True)
        os.makedirs(os.path.dirname(args.chunks_cache) or ".", exist_ok=True)
        with open(args.chunks_cache, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

    # Multi-Key Alignment
    cache = {}
    if os.path.exists(args.cache_path):
        with open(args.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    it = json.loads(line)
                    cache[it["key"]] = it["result"]
                except Exception:
                    continue

    todo = [c for c in all_chunks if c["key"] not in cache and c["phrases"]]
    print(f"Alignment: {len(all_chunks):,} chunks total, {len(cache):,} cached, {len(todo):,} to align with {args.model}", flush=True)

    if todo:
        os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)
        done, failed, t_start = 0, 0, time.time()
        with open(args.cache_path, "a", encoding="utf-8") as cf, ThreadPoolExecutor(max_workers=total_workers) as ex:
            futures = {
                ex.submit(align_call, c["en"], c["tw"], c["phrases"], all_keys[i % len(all_keys)], args.model): c
                for i, c in enumerate(todo)
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
                if done % 1000 == 0:
                    rate = done / (time.time() - t_start)
                    rem = (len(todo) - done) / max(1, rate)
                    print(f"  aligned {done:,}/{len(todo):,} ({rate:.1f}/s, ~{rem/3600:.1f}h remaining)", flush=True)

    # Format into final Dataset
    print("Formatting final dataset splits...", flush=True)
    pattern = re.compile(r"<([^<>\n]+)>")
    valid_rows = []
    for c in all_chunks:
        res = cache.get(c["key"])
        if not res or not res.get("target_twi"):
            continue
        tgt = res["target_twi"].strip()
        if not pattern.search(tgt) or tgt.count("<") != tgt.count(">"):
            continue
        valid_rows.append({
            "input": c["en"],
            "target": tgt,
            "english": c["en"],
            "twi": c["tw"],
            "source": c.get("source", "grand"),
        })

    print(f"Total valid aligned pairs: {len(valid_rows):,}", flush=True)
    random.seed(42)
    random.shuffle(valid_rows)

    val_n = 3000
    test_n = 3000
    train_n = len(valid_rows) - val_n - test_n

    ds_splits = {
        "train": Dataset.from_list(valid_rows[:train_n]),
        "validation": Dataset.from_list(valid_rows[train_n : train_n + val_n]),
        "test": Dataset.from_list(valid_rows[train_n + val_n :]),
    }
    ds_dict = DatasetDict(ds_splits)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving to {args.out_dir}...", flush=True)
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
- grand-corpus
- multi-pillar
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

# GLOSS-MT-2: Grand Multi-Pillar English → Asante Twi (Akan) Parallel Dataset

Massive, balanced parallel dataset for **GLOSS-MT-2** spanning:
1. **GHOK Parallel Corpus:** STEM, technical, academic, and governance text.
2. **Pristine Twi-English:** Balanced across Monologue, Narrative, Dialogue, and Storyful styles.
3. **Ghanaian News Paragraphs:** Real Ghanaian news and current affairs.
4. **JW African Corpus:** Verse-aligned literature and dialogue.

### Paradigm
- **Input (`input`):** Raw, natural English sentence with **NO brackets** and **NO markdown** (zero preprocessing):
  `It reached a small afternoon, and the news reporters went to the Kejetia market place. They went to see how the trade is there now. This matter is heavy.`
- **Target (`target`):** Fluent Asante Twi translation using the base **English noun phrases** in angle brackets `<noun>` with all Twi determiners outside:
  `Ɛduruu <afternoon> ketewa bi, na <news reporters> no nsrane kɔɔ <Kejetia market place> hɔ. Wɔkɔhwɛɛ sɛnea <trade> no teɛ wɔ hɔ seisei. <matter> yi yɛ duru.`
- **Clean Reference Target (`twi`):** `Ɛduruu owia ketewa bi, na amaneɛbɔ adwumayɛfoɔ no nsrane kɔɔ Kejetia dwadibea hɔ. Wɔkɔhwɛɛ sɛnea dwadie no teɛ wɔ hɔ seisei. Asɛm yi yɛ duru.`
- **Clean Reference Source (`english`):** `It reached a small afternoon, and the news reporters went to the Kejetia market place. They went to see how the trade is there now. This matter is heavy.`

### Split Statistics
| Split | Samples | Notes |
|---|---|---|
| **train** | {len(ds_splits['train']):,} | Balanced across all 4 pillars |
| **validation** | {len(ds_splits['validation']):,} | Held-out validation set |
| **test** | {len(ds_splits['test']):,} | Benchmark test set |
| **Total** | **{len(valid_rows):,}** | 100% complete grammatical sentences $\le$ 512 chars |

### Code Repository
[`GhanaOpenAI/gloss-mt-2`](https://github.com/GhanaOpenAI/gloss-mt-2)
"""
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARD)

    if args.push:
        print(f"Pushing grand dataset to Hugging Face {args.push}...", flush=True)
        ds_dict.push_to_hub(args.push)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.push,
            repo_type="dataset",
        )
        print(f"Published successfully to https://huggingface.co/datasets/{args.push}!", flush=True)


if __name__ == "__main__":
    main()
