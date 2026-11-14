"""Selective Noun Phrase & Named Entity Extraction for GLOSS-MT-2.

Rules:
1. Target multi-word noun phrases (>= 2 words, where the second word can be a number, e.g. 'May 2026', 'Vision 2030', 'news reporters', 'high court').
2. Target single-word nouns ONLY if capitalized (i.e. Named Entities / Proper Nouns, e.g. 'Ghana', 'Telecel', 'Jesus').
3. Common single-word lowercase nouns ('man', 'day', 'matter', 'soldier') are left unbracketed so models translate them freely and naturally.
"""

import re
from typing import List, Optional

_nlp = None
_LEADING_ARTICLES = re.compile(r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE)


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


def extract_selective_noun_phrases(text: str, nlp=None) -> List[str]:
    """Extract multi-word noun phrases (>=2 words, numbers allowed) and capitalized single-word entities."""
    nlp = nlp or get_nlp()
    if nlp is None:
        words = [w for w in re.findall(r"\b[A-Z][a-zA-Z0-9-]*\b", text) if len(w) >= 2]
        return list(dict.fromkeys(words))

    doc = nlp(text)
    phrases = []
    seen = set()

    raw_spans = []
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PRON":
            continue
        start, end = chunk.start, chunk.end
        # If the token immediately following is a number (e.g. "May 2026", "Vision 2030"), extend span
        if end < len(doc) and (doc[end].like_num or doc[end].pos_ == "NUM"):
            end += 1
        raw_spans.append((start, end))

    for start, end in raw_spans:
        span_text = doc[start:end].text.strip()
        span_clean = re.sub(r"^[\W_]+|[\W_]+$", "", span_text)
        base = _LEADING_ARTICLES.sub("", span_clean).strip()
        words = base.split()
        if not words:
            continue

        # Criteria:
        # 1. Multi-word phrase (>= 2 words)
        # 2. Single-word ONLY if capitalized (Named Entity / Proper Noun)
        is_valid = False
        if len(words) >= 2:
            is_valid = True
        elif len(words) == 1:
            w = words[0]
            if w[:1].isupper() and any(c.isalpha() for c in w):
                is_valid = True

        if is_valid and len(base) >= 2 and base.lower() not in seen:
            seen.add(base.lower())
            phrases.append(base)

    return phrases


def extract_noun_phrases(text: str, lang: str = "en", **kwargs) -> List[str]:
    return extract_selective_noun_phrases(text, **kwargs)
