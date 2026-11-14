"""Selective Noun Phrase & Named Entity Extraction for GLOSS-MT-2.

Linguistic & Grammatical Rules:
1. Full Grammatical Nominal Phrases:
   Rather than shallow flat base-noun chunking (which artificially chops nominal complements,
   e.g. splitting 'president of Ghana' into 'president' and 'Ghana'), extraction follows the full
   syntactic dependency tree to capture complete complex nominal phrases:
   - 'president of Ghana', 'Ministry of Finance', 'Bank of Ghana governor', 'King of Asante', 'Minister for Health'
2. Abstract Nouns are EXCLUDED (translated freely into Twi):
   Abstract nouns and conceptual terms ('information', 'development', 'history', 'investigation',
   'peace', 'justice', 'freedom', 'policy', 'decision', 'beauty', 'knowledge') are NEVER bracketed.
   Models translate them naturally into fluent Asante Twi.
3. Concrete (Non-Abstract) Noun Phrases are INCLUDED (even if 1 word):
   Physical entities, artifacts, locations, organisms, and institutions
   ('hospital', 'regional hospital', 'car', 'road', 'soldier', 'gold', 'water', 'phone', 'court', 'market place').
4. Capitalized Nouns & Proper Entities:
   - Capitalized entities inside the sentence are preserved ('Ghana', 'Telecel', 'Ministry of Finance', 'May 2026').
   - Sentence-Initial Capitalization Guard:
     - Multi-word capitalized phrases starting the sentence are preserved ('Ministry of Finance...').
     - Single-word nouns starting the sentence ('Information...', 'Hospital...') are subject to the
       abstractness test: kept if concrete ('Hospital...'), excluded if abstract ('Information...').
"""

import re
from typing import List, Optional, Set

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
    "police", "army", "family", "team", "equipment", "device", "vehicle",
    "governor", "president", "king", "minister", "chief", "director"
}

NOMINAL_DEPS = {"compound", "amod", "nmod", "nummod", "det", "poss", "case", "appos"}


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


def is_relational_prep(prep_token) -> bool:
    """Returns True if preposition is a syntactic complement of a nominal phrase."""
    # 'of' is universally a nominal genitive/relational complement
    if prep_token.text.lower() == "of":
        return True
    # 'for' or 'to' when connecting capitalized title entities (e.g. 'Minister for Health')
    if prep_token.head.text[:1].isupper() and any(c.text[:1].isupper() for c in prep_token.children if c.dep_ in ("pobj", "pcomp")):
        if prep_token.text.lower() in ("for", "to"):
            return True
    return False


def get_nlp(model_name: str = "en_core_web_sm"):
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(model_name, disable=["ner"])
        except Exception:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except Exception:
                return None
    return _nlp


def extract_selective_noun_phrases(text: str, nlp=None) -> List[str]:
    """
    Extracts full grammatical noun phrases (including prepositional complements like 'president of Ghana'),
    filtering out abstract nouns while preserving concrete entities and proper nouns.
    """
    nlp = nlp or get_nlp()
    if nlp is None:
        words = [w for w in re.findall(r"\b[A-Z][a-zA-Z0-9-]*\b", text) if len(w) >= 2]
        return list(dict.fromkeys(words))

    doc = nlp(text)

    # 1. Identify top-level nominal heads in dependency tree
    heads = []
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            if token.dep_ in ("compound", "nmod"):
                continue
            # If it's a pobj of a relational preposition ('of Ghana'), its parent noun absorbs it
            if token.dep_ in ("pobj", "pcomp") and is_relational_prep(token.head) and token.head.head.pos_ in ("NOUN", "PROPN"):
                continue
            heads.append(token)

    selected = []
    seen = set()

    # 2. Collect full nominal subtrees for each head
    for head in heads:
        nodes = {head}

        def collect(node):
            for child in node.children:
                if child.dep_ in NOMINAL_DEPS:
                    nodes.add(child)
                    collect(child)
                elif child.dep_ == "prep" and is_relational_prep(child):
                    nodes.add(child)
                    for pobj in child.children:
                        if pobj.dep_ in ("pobj", "pcomp"):
                            nodes.add(pobj)
                            collect(pobj)

        collect(head)
        sorted_nodes = sorted(list(nodes), key=lambda t: t.i)
        span_text = doc[sorted_nodes[0].i : sorted_nodes[-1].i + 1].text.strip()
        span_clean = re.sub(r"^[\W_]+|[\W_]+$", "", span_text)
        base = _LEADING_ARTICLES.sub("", span_clean).strip()
        if not base or len(base) < 2:
            continue

        words = base.split()

        # Check sentence start
        m = re.search(r"\b" + re.escape(base) + r"\b", text)
        start_pos = m.start() if m else 0
        prefix = text[:start_pos].strip(" \"'`([{<-")
        is_sent_start = (len(prefix) == 0)

        # Capitalization check
        has_capitalized = any(w[:1].isupper() and any(c.isalpha() for c in w) for w in words)

        # Root abstractness check
        root_word = head.text
        is_abstract = is_abstract_word(root_word)

        # Apply user selection rules:
        keep = False
        if has_capitalized:
            if is_sent_start:
                if len(words) > 1:
                    # Multi-word capitalized phrase starting sentence (e.g. 'Ministry of Finance')
                    keep = True
                else:
                    # Single word capitalized at sentence start (e.g. 'Information' vs 'Hospital')
                    keep = not is_abstract
            else:
                # Capitalized anywhere else in sentence: keep regardless of abstractness
                keep = True
        else:
            # Lowercase: keep if not abstract
            keep = not is_abstract

        if keep and base.lower() not in seen:
            seen.add(base.lower())
            selected.append(base)

    return selected


def extract_noun_phrases(text: str, lang: str = "en", **kwargs) -> List[str]:
    return extract_selective_noun_phrases(text, **kwargs)
