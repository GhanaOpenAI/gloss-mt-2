"""Noun phrase extraction for source texts.

For English: extracts noun chunks via spaCy (nkrane-style), stripping stopwords and punctuation.
For Twi: extracts multi-word and content terms using vocabulary lookups and morphological patterns.
"""

import re
from typing import List, Optional

_nlp = None


def get_nlp(model_name: str = "en_core_web_sm"):
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
        except Exception:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except Exception:
                return None
    return _nlp


def extract_noun_phrases_en(text: str, nlp=None) -> List[str]:
    """Extract key noun phrases from English text."""
    nlp = nlp or get_nlp()
    if nlp is None:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return list(dict.fromkeys(words))

    doc = nlp(text)
    noun_phrases: List[str] = []
    seen = set()

    for chunk in doc.noun_chunks:
        tokens = list(chunk)
        if all(token.is_stop for token in tokens):
            continue

        content_tokens = [token for token in tokens if not token.is_stop and not token.is_punct]
        if not content_tokens:
            continue

        raw = " ".join(t.text for t in content_tokens).strip().lower()
        cleaned = re.sub(r"\s*-\s*", "-", raw)
        cleaned = re.sub(r"\s*['’]\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if len(cleaned) >= 2 and cleaned not in seen and re.search(r"[a-z]{2,}", cleaned):
            seen.add(cleaned)
            noun_phrases.append(cleaned)

    return noun_phrases


def extract_noun_phrases_tw(text: str, known_vocab: Optional[set] = None) -> List[str]:
    """Extract candidate noun phrases from Twi text using word boundaries and known vocabulary."""
    words = re.findall(r"\b[a-zA-ZɛɔƐƆ'-]{2,}\b", text)
    seen, phrases = set(), []
    for w in words:
        wl = w.lower()
        if known_vocab is not None and wl not in known_vocab:
            continue
        if wl not in seen and len(wl) >= 3:
            seen.add(wl)
            phrases.append(w)
    return phrases


def extract_noun_phrases(text: str, lang: str = "en", **kwargs) -> List[str]:
    if lang.lower().startswith("en"):
        return extract_noun_phrases_en(text, **kwargs)
    elif lang.lower().startswith("tw"):
        return extract_noun_phrases_tw(text, **kwargs)
    return extract_noun_phrases_en(text, **kwargs)
