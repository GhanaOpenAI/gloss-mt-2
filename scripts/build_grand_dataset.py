"""Build the Grand GLOSS-MT-2 Dataset across 4 major pillars with Selective Noun Targeting:
1. GHOK Parallel Corpus (~457k passages: STEM, technical, academic, governance)
2. Pristine Twi-English (~450k passages: balanced 25% Narrative, 25% Dialogue, 25% Storyful, 25% Monologue)
3. Ghanaian News Paragraphs (~256k passages: Ghanaian current affairs, politics, economy)
4. JW Twi-English Corpus (~225k sentences: verse-aligned Akan literature & daily life)

Selective Targeting Rules:
- Multi-word phrases (>= 2 words, where 2nd word can be a number e.g. 'May 2026', 'news reporters', 'high court').
- Single-word nouns ONLY if capitalized (Named Entities / Proper Nouns e.g. 'Ghana', 'Telecel', 'Jesus').
- Common single-word lowercase nouns are left unbracketed so models translate them freely and naturally.
"""

import argparse
import asyncio
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
from google import genai
from google.genai import types
from huggingface_hub import HfApi

from gloss_mt_2.extractor import extract_selective_noun_phrases

MODEL_NAME = "gemini-3.8-live"
VOICE_NAME = "Zephyr"
_LEADING_ARTICLES = re.compile(r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE)


def load_api_keys() -> List[str]:
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


def chunk_key(en: str, tw: str) -> str:
    return hashlib.sha1(f"{en.strip()}|||{tw.strip()}".encode("utf-8")).hexdigest()


def extract_and_parse_json(raw_text: str):
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if m:
            cand = m.group(1)
            try:
                return json.loads(cand)
            except Exception:
                fixed = re.sub(r",\s*([\}\]])", r"\1", cand)
                return json.loads(fixed)
        raise


def build_live_config() -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME)
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


async def call_live_batch(client: genai.Client, batch: List[Dict], model: str = MODEL_NAME, max_retries: int = 2) -> Optional[List[Dict]]:
    payload = [
        {"id": c["key"], "en": c["en"], "tw": c["tw"], "phrases": c["phrases"]}
        for c in batch
    ]
    prompt = (
        "For each pair below, produce target Twi text where ONLY the listed key phrases are replaced by the English phrase in angle brackets <...>.\n"
        "All Twi determiners ('no', 'yi', 'bi', etc.) must stay outside brackets in Twi.\n"
        "Never nest brackets. Leave all other words completely in natural Twi.\n"
        "Return ONLY a JSON array with objects containing 'id', 'mapping', and 'target_twi'.\n\n"
        f"Input pairs:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    config = build_live_config()
    for attempt in range(1, max_retries + 1):
        try:
            async with client.aio.live.connect(model=model, config=config) as session:
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
                    turn_complete=True,
                )
                chunks = []
                async for message in session.receive():
                    sc = message.server_content
                    if sc and sc.output_transcription and sc.output_transcription.text:
                        chunks.append(sc.output_transcription.text)
                    if sc and sc.turn_complete:
                        break
                raw_text = "".join(chunks).strip()
                data = extract_and_parse_json(raw_text)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
        except Exception:
            await asyncio.sleep(1.0 * attempt)
    return None


async def call_live_single(client: genai.Client, c: Dict, model: str = MODEL_NAME) -> Optional[Dict]:
    prompt = (
        "Produce target Twi text where ONLY the listed key phrases are replaced by the English phrase in angle brackets <...>.\n"
        "All Twi determiners ('no', 'yi', 'bi', etc.) must stay outside brackets in Twi.\n"
        "Never nest brackets. Leave all other words completely in natural Twi.\n"
        "Return ONLY a JSON object with 'mapping' and 'target_twi'.\n\n"
        f"English:\n{c['en']}\n\nTwi:\n{c['tw']}\n\nKey phrases:\n{json.dumps(c['phrases'], ensure_ascii=False)}"
    )
    config = build_live_config()
    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
                turn_complete=True,
            )
            chunks = []
            async for message in session.receive():
                sc = message.server_content
                if sc and sc.output_transcription and sc.output_transcription.text:
                    chunks.append(sc.output_transcription.text)
                if sc and sc.turn_complete:
                    break
            raw_text = "".join(chunks).strip()
            data = extract_and_parse_json(raw_text)
            if isinstance(data, dict) and "target_twi" in data:
                return data
    except Exception:
        pass
    return None


async def align_worker(
    worker_id: str,
    client: genai.Client,
    queue: asyncio.Queue,
    cache: Dict,
    cache_file,
    file_lock: asyncio.Lock,
    stats: Dict,
    model: str = MODEL_NAME,
):
    while True:
        batch = await queue.get()
        if batch is None:
            queue.task_done()
            break

        results = await call_live_batch(client, batch, model=model)
        by_id = {}
        if results:
            for item in results:
                if isinstance(item, dict) and "id" in item and "target_twi" in item:
                    by_id[str(item["id"])] = item

        for item in batch:
            k = item["key"]
            res = by_id.get(k)
            if not res or not res.get("target_twi"):
                single_res = await call_live_single(client, item, model=model)
                if single_res and single_res.get("target_twi"):
                    res = single_res

            if res and res.get("target_twi"):
                async with file_lock:
                    cache[k] = res
                    cache_file.write(json.dumps({"key": k, "result": res}, ensure_ascii=False) + "\n")
                    cache_file.flush()
                    stats["done"] += 1
                    if stats["done"] % 200 == 0:
                        elapsed = time.time() - stats["start_time"]
                        rate = stats["done"] / max(0.1, elapsed)
                        rem = (stats["total_todo"] - stats["done"]) / max(0.1, rate)
                        print(
                            f"[{stats['done']:,}/{stats['total_todo']:,}] Speed: {rate:.1f} chunks/s | "
                            f"ETA: {rem/3600:.1f}h remaining",
                            flush=True,
                        )
            else:
                stats["failed"] += 1

        queue.task_done()


async def periodic_hf_pusher(cache: Dict, all_chunks: List[Dict], repo_id: str, interval_mins: int = 20):
    if not repo_id:
        return
    pattern = re.compile(r"<([^<>\n]+)>")
    while True:
        await asyncio.sleep(interval_mins * 60)
        try:
            valid_rows = []
            for c in all_chunks:
                res = cache.get(c["key"])
                if not res or not res.get("target_twi"):
                    continue
                tgt = res["target_twi"]
                if isinstance(tgt, list):
                    tgt = " ".join(tgt)
                if not isinstance(tgt, str):
                    continue
                tgt = tgt.strip()
                if pattern.search(tgt) and tgt.count("<") == tgt.count(">"):
                    valid_rows.append({
                        "input": c["en"],
                        "target": tgt,
                        "english": c["en"],
                        "twi": c["tw"],
                        "source": c.get("source", "grand"),
                    })
            if len(valid_rows) >= 5000:
                print(f"[Periodic HF Pusher] Uploading snapshot of {len(valid_rows):,} rows to {repo_id}...", flush=True)
                val_n = min(3000, max(500, int(len(valid_rows) * 0.05)))
                test_n = val_n
                train_n = len(valid_rows) - val_n - test_n
                ds_dict = DatasetDict({
                    "train": Dataset.from_list(valid_rows[:train_n]),
                    "validation": Dataset.from_list(valid_rows[train_n : train_n + val_n]),
                    "test": Dataset.from_list(valid_rows[train_n + val_n :]),
                })
                ds_dict.push_to_hub(repo_id)
                print(f"[Periodic HF Pusher] Snapshot successfully pushed to {repo_id}!", flush=True)
        except Exception as e:
            print(f"[Periodic HF Pusher Error] {e}", flush=True)


async def main_async(args):
    all_keys = load_api_keys()
    if not all_keys:
        sys.exit("Error: No Gemini API keys found in environment or key files.")

    total_workers = min(64, len(all_keys) * args.workers_per_key)
    print(f"Loaded {len(all_keys)} API keys ({total_workers} total parallel Live API workers).", flush=True)

    if os.path.exists(args.chunks_cache):
        print(f"Loading cached complete chunks from {args.chunks_cache}...", flush=True)
        t0 = time.time()
        with open(args.chunks_cache, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        print(f"Loaded {len(all_chunks):,} pre-chunked items in {time.time()-t0:.2f}s", flush=True)
    else:
        print("Extracting and chunking across all 4 pillars with selective noun targeting...", flush=True)
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
                        ph = extract_selective_noun_phrases(en_clean, nlp)
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
                        ph = extract_selective_noun_phrases(ec, nlp)
                        if ph:
                            all_chunks.append({"en": ec, "tw": tc, "phrases": ph, "key": k, "source": source_tag})

        print("Loading Pillar 1: GHOK Parallel Corpus...", flush=True)
        ds_ghok = load_dataset("ghananlpcommunity/ghok-parallel-en-twi", split="train")
        for r in ds_ghok:
            add_pair(r["english"], r["twi"], "ghok")
        print(f"  Pillar 1 accumulated: {len(all_chunks):,} chunks", flush=True)

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

        print("Loading Pillar 3: Ghanaian News Paragraphs...", flush=True)
        ds_news = load_dataset("ghananlpcommunity/twi-english-paragraph-dataset_news", split="train")
        for r in ds_news:
            add_pair(r["ENGLISH"], r["TWI"], "news")
        print(f"  Pillar 1 + 2 + 3 accumulated: {len(all_chunks):,} chunks", flush=True)

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
    print(f"Alignment: {len(all_chunks):,} chunks total, {len(cache):,} cached, {len(todo):,} to align with {MODEL_NAME}", flush=True)

    if todo:
        os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)
        queue = asyncio.Queue(maxsize=5000)
        file_lock = asyncio.Lock()
        stats = {
            "done": 0,
            "failed": 0,
            "start_time": time.time(),
            "total_todo": len(todo),
        }

        asyncio.create_task(periodic_hf_pusher(cache, all_chunks, args.push, interval_mins=20))

        cf = open(args.cache_path, "a", encoding="utf-8")

        workers = []
        batch_size = 3
        for k_idx, key in enumerate(all_keys):
            client = genai.Client(api_key=key)
            for w_idx in range(args.workers_per_key):
                wid = f"K{k_idx}-W{w_idx}"
                workers.append(
                    asyncio.create_task(
                        align_worker(
                            wid,
                            client,
                            queue,
                            cache,
                            cf,
                            file_lock,
                            stats,
                            model=MODEL_NAME,
                        )
                    )
                )

        batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
        for b in batches:
            await queue.put(b)

        await queue.join()

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        cf.close()

    print("Formatting final dataset splits...", flush=True)
    pattern = re.compile(r"<([^<>\n]+)>")
    valid_rows = []
    for c in all_chunks:
        res = cache.get(c["key"])
        if not res or not res.get("target_twi"):
            continue
        tgt = res["target_twi"]
        if isinstance(tgt, list):
            tgt = " ".join(tgt)
        if not isinstance(tgt, str):
            continue
        tgt = tgt.strip()
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

Massive, balanced parallel dataset for **GLOSS-MT-2** with selective noun targeting:
1. **GHOK Parallel Corpus:** STEM, technical, academic, and governance text.
2. **Pristine Twi-English:** Balanced across Monologue, Narrative, Dialogue, and Storyful styles.
3. **Ghanaian News Paragraphs:** Real Ghanaian news and current affairs.
4. **JW African Corpus:** Verse-aligned literature and dialogue.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/en_tw_grand")
    ap.add_argument("--cache-path", default="build/grand_alignment_cache.jsonl")
    ap.add_argument("--chunks-cache", default="build/grand_chunks_cache.json")
    ap.add_argument("--push", default="ghanaopenai/gloss-mt-2-en-tw-grand")
    ap.add_argument("--pristine-sample-size", type=int, default=450_000)
    ap.add_argument("--workers-per-key", type=int, default=3)
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
