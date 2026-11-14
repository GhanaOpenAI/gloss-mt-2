"""Selective Noun Phrase & Named Entity Extraction for GLOSS-MT-2.

Rules:
1. Leave out abstract nouns (e.g. 'information', 'development', 'history', 'investigation', 'peace', 'justice').
   Models must translate abstract concepts freely and naturally into Twi.
2. If a noun phrase is NOT abstract (concrete/physical: e.g. 'hospital', 'car', 'road', 'soldier', 'gold'),
   include it even if it is 1 word!
3. If a noun phrase is capitalized:
   - If it appears anywhere except the sentence start: KEEP it irrespective of abstract or not
     (e.g. 'Ghana', 'Telecel', 'Ministry of Finance', 'May 2026', 'Christianity').
   - If it appears at the start of the sentence:
     - If it contains MORE THAN ONE word (e.g. 'Ministry of Finance', 'Telecel Ghana'): KEEP it!
     - If it is a SINGLE word (e.g. 'Information...', 'Development...', 'Hospital...'):
       capitalization may just be grammatical sentence-initial casing, so apply the abstract rule:
       keep ONLY if NOT abstract!
"""

import re
from typing import List, Optional

_nlp = None
_wn = None
_LEADING_ARTICLES = re.compile(r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE)

CONCRETE_LEXNAMES = {
    "noun.artifact", "noun.person", "noun.location", "noun.plant",
    "noun.animal", "noun.substance", "noun.food", "noun.body", "noun.object",
    "noun.group"
}

ABSTRACT_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence",
    "ship", "hood", "dom", "ism", "cracy", "logy", "phy",
    "ure"
)

CONCRETE_EXCEPTIONS = {
    "instrument", "monument", "apartment", "document", "garment", "settlement",
    "building", "painting", "drawing", "clothing", "ring", "wing", "spring",
    "gold", "silver", "money", "court", "market", "office", "church", "school",
    "hospital", "bank", "prison", "station", "university", "college",
    "police", "army", "family", "team", "equipment", "device", "vehicle"
}


def get_wordnet():
    global _wn
    if _wn is None:
        try:
            import nltk
            try:
                from nltk.corpus import wordnet as wn
                _ = wn.synsets("test")
                _wn = wn
            except Exception:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                from nltk.corpus import wordnet as wn
                _wn = wn
        except Exception:
            return None
    return _wn


def is_abstract_word(word: str) -> bool:
    """Returns True if the head noun is an abstract concept rather than a concrete entity."""
    lower = word.lower()
    if lower in CONCRETE_EXCEPTIONS:
        return False

    wn = get_wordnet()
    if wn is not None:
        syns = wn.synsets(lower, pos=wn.NOUN)
        if syns:
            if any(s.lexname() in CONCRETE_LEXNAMES for s in syns[:3]):
                if syns[0].lexname() in {"noun.attribute", "noun.feeling", "noun.state", "noun.motive"}:
                    return True
                return False
            return True

    return any(lower.endswith(sfx) for sfx in ABSTRACT_SUFFIXES)


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
    """
    Extracts non-abstract concrete noun phrases and legitimate capitalized entities.
    Abstract nouns are excluded so models translate them freely and naturally.
    """
    nlp = nlp or get_nlp()
    if nlp is None:
        words = [w for w in re.findall(r"\b[A-Z][a-zA-Z0-9-]*\b", text) if len(w) >= 2]
        return list(dict.fromkeys(words))

    doc = nlp(text)
    phrases = []
    seen = set()

    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PRON":
            continue

        # Split coordinate conjunctions (e.g. 'early development and hospital equipment')
        raw_parts = re.split(r"\s+\b(?:and|or)\b\s+", chunk.text)
        for part in raw_parts:
            part_clean = re.sub(r"^[\W_]+|[\W_]+$", "", part.strip())
            base = _LEADING_ARTICLES.sub("", part_clean).strip()
            if not base or len(base) < 2:
                continue

            words = base.split()
            # The root/head word of the noun phrase is typically the last token
            root_word = chunk.root.text if (len(words) > 1 and chunk.root.text in words) else words[-1]
            is_abstract = is_abstract_word(root_word)

            # Determine whether phrase begins at sentence start (ignoring leading quotes/brackets)
            m = re.search(r"\b" + re.escape(base) + r"\b", text)
            start_pos = m.start() if m else 0
            prefix = text[:start_pos].strip(" \"'`([{<-")
            is_sent_start = (len(prefix) == 0)

            is_capitalized = words[0][:1].isupper() and any(c.isalpha() for c in words[0])

            # Apply user rules:
            keep = False
            if is_capitalized:
                if is_sent_start:
                    if len(words) > 1:
                        # Multi-word capitalized phrase starting the sentence (e.g. 'Ministry of Finance')
                        keep = True
                    else:
                        # Single word capitalized at sentence start (e.g. 'Information' vs 'Hospital')
                        keep = not is_abstract
                else:
                    # Capitalized anywhere else in sentence: keep regardless of abstract or not
                    keep = True
            else:
                # Lowercase: keep if not abstract
                keep = not is_abstract

            if keep and base.lower() not in seen:
                seen.add(base.lower())
                phrases.append(base)

    return phrases


def extract_noun_phrases(text: str, lang: str = "en", **kwargs) -> List[str]:
    return extract_selective_noun_phrases(text, **kwargs)
