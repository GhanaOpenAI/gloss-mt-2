"""Build parallel datasets for GLOSS-MT-2 where source noun phrases are preserved verbatim in the target."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gloss_mt_2.lookup import TwiLookup, DEFAULT_PATH


_SPLIT = re.compile(r"(\n\s*)")


def split_lines(text: str) -> Tuple[List[str], List[str]]:
    parts = _SPLIT.split(text)
    lines = [parts[i] for i in range(0, len(parts), 2)]
    seps = [parts[i] for i in range(1, len(parts), 2)] + [""]
    return lines, seps


def join_span(lines: List[str], seps: List[str], start: int, end: int) -> str:
    return "".join(lines[i] + seps[i] for i in range(start, end))


def pack_lines(lengths: List[int], max_tokens: int) -> List[Tuple[int, int]]:
    spans, start, cur = [], 0, 0
    for i, n in enumerate(lengths):
        if cur + n > max_tokens and cur > 0:
            spans.append((start, i))
            start, cur = i, 0
        cur += n
    if cur > 0 or not spans:
        spans.append((start, len(lengths)))
    return spans


def chunk_key(tw: str, phrases: List[str]) -> str:
    return hashlib.sha1(json.dumps([tw, phrases], ensure_ascii=False).encode("utf-8")).hexdigest()


def build_preserved_targets(
    en_text: str,
    tw_bracketed: str,
    mapping: Dict[str, Optional[str]],
) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    1. en_to_tw_target: Twi text where each <twi_span> is replaced by its English noun phrase.
    2. tw_to_en_target: English text where each english_phrase is replaced by its Twi span.
    """
    twi_to_en = {}
    valid_pairs = []
    for en_p, tw_s in mapping.items():
        if tw_s and isinstance(tw_s, str) and tw_s.strip():
            ep = en_p.strip()
            ts = tw_s.strip()
            twi_to_en[ts.lower()] = ep
            valid_pairs.append((ep, ts))

    # 1. en -> tw target (Twi with English noun phrases)
    def sub_tw(m):
        span = m.group(1).strip()
        en_phrase = twi_to_en.get(span.lower(), span)
        if span[:1].isupper() and not en_phrase[:1].isupper():
            en_phrase = en_phrase[:1].upper() + en_phrase[1:]
        return en_phrase

    tw_preserved = re.sub(r"<([^<>\n]+)>", sub_tw, tw_bracketed)

    # 2. tw -> en target (English with Twi noun phrases)
    valid_pairs.sort(key=lambda x: len(x[0]), reverse=True)
    table = {k.lower(): v for k, v in valid_pairs}
    if table:
        alts = [r"\s+".join(map(re.escape, k.split())) for k in sorted(table.keys(), key=len, reverse=True)]
        pat = re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE)

        def sub_en(m):
            matched = m.group(0)
            k = " ".join(matched.lower().split())
            twi_val = table.get(k, matched)
            if matched[:1].isupper() and not twi_val[:1].isupper():
                twi_val = twi_val[:1].upper() + twi_val[1:]
            return twi_val

        en_preserved = pat.sub(sub_en, en_text)
    else:
        en_preserved = en_text

    used_terms = [{"english": p, "twi": s} for p, s in valid_pairs]
    return tw_preserved, en_preserved, used_terms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", help="Directory containing train.jsonl, validation.jsonl, test.jsonl")
    ap.add_argument("--align-cache", required=True, help="Path to gold_alignment_cache.jsonl")
    ap.add_argument("--out-dir", default="build/preserved-dataset", help="Output directory for generated datasets")
    ap.add_argument("--chunk-tokens", type=int, default=192)
    ap.add_argument("--push", default=None, help="Hugging Face repo id to push to (optional)")
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer
    from gloss_mt_2.extractor import extract_noun_phrases

    print(f"Loading alignment cache from {args.align_cache}...")
    t0 = time.time()
    cache = {}
    with open(args.align_cache, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            cache[it["key"]] = it.get("result", it.get("mapping"))
    print(f"Loaded {len(cache):,} alignments in {time.time()-t0:.2f}s")

    tok = AutoTokenizer.from_pretrained("google/umt5-xl")

    splits_en2tw = {}
    splits_tw2en = {}

    for split in ("train", "validation", "test"):
        split_path = os.path.join(args.data, f"{split}.jsonl")
        if not os.path.exists(split_path):
            print(f"Skipping {split} (file not found: {split_path})")
            continue

        with open(split_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        en2tw_rows = []
        tw2en_rows = []
        skipped = 0

        for r in rows:
            en_lines, en_seps = split_lines(r["english"])
            tw_lines, tw_seps = split_lines(r["translation"])
            if len(en_lines) != len(tw_lines):
                skipped += 1
                continue

            lengths = [len(tok(l, add_special_tokens=False).input_ids) for l in en_lines]
            spans = pack_lines(lengths, args.chunk_tokens)

            for i, (s, e) in enumerate(spans):
                en = join_span(en_lines, en_seps, s, e)
                tw = join_span(tw_lines, tw_seps, s, e)
                phrases = extract_noun_phrases(en, lang="en")
                key = chunk_key(tw, phrases)
                al = cache.get(key)

                if al and isinstance(al, dict) and al.get("bracketed_twi") and al.get("mapping"):
                    tw_pres, en_pres, terms = build_preserved_targets(en, al["bracketed_twi"], al["mapping"])
                else:
                    tw_pres = tw
                    en_pres = en
                    terms = []

                doc_id = r.get("id", f"{split}_{len(en2tw_rows)}")

                en2tw_rows.append({
                    "id": doc_id,
                    "chunk_index": i,
                    "source": en,
                    "target": tw_pres,
                    "gold": tw,
                    "terms": terms,
                })

                tw2en_rows.append({
                    "id": doc_id,
                    "chunk_index": i,
                    "source": tw,
                    "target": en_pres,
                    "gold": en,
                    "terms": terms,
                })

        print(f"{split}: {len(rows) - skipped:,} docs -> {len(en2tw_rows):,} chunks (skipped {skipped})")
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

    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
