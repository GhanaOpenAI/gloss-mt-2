"""Post-translation glossary substitution and inference pipeline for GLOSS-MT-2."""

import re
from typing import Dict, List, Optional, Tuple

from .extractor import extract_noun_phrases
from .lookup import TwiLookup, normalize_term


def substitute_glossary(
    translation: str,
    source_terms: List[str],
    lookup: Optional[TwiLookup] = None,
    direction: str = "en-tw",
    custom_glossary: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Detect preserved source terms inside the translated output and replace them
    with their corresponding glossary translations.
    """
    if not source_terms:
        return translation, []

    # Combine custom glossary and lookup
    active_dict = {}
    for term in source_terms:
        norm = normalize_term(term)
        trans = None
        if custom_glossary:
            # Check custom glossary first
            for k, v in custom_glossary.items():
                if normalize_term(k) == norm:
                    trans = v
                    break
        if trans is None and lookup is not None:
            trans = lookup.lookup(norm, direction=direction)

        if trans:
            active_dict[norm] = trans

    if not active_dict:
        return translation, []

    # Sort terms by length descending to match longest phrases first
    # Map normalized term back to candidate regex patterns
    sorted_terms = sorted(source_terms, key=len, reverse=True)
    applied = []
    out = translation

    for term in sorted_terms:
        norm = normalize_term(term)
        if norm not in active_dict:
            continue
        target_trans = active_dict[norm]

        words = term.split()
        if not words:
            continue

        # Regex matching whole words, ignoring case
        pattern = re.compile(
            r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)",
            re.IGNORECASE,
        )

        def sub_fn(m):
            matched = m.group(0)
            res = target_trans
            # Preserve case
            if matched.isupper() and len(matched) > 1:
                res = res.upper()
            elif matched[:1].isupper() and not res[:1].isupper():
                res = res[:1].upper() + res[1:]
            applied.append({"source": matched, "target": res})
            return res

        out = pattern.sub(sub_fn, out)

    return out, applied


class GlossMT2Translator:
    """End-to-end translator using models trained to preserve source nouns."""

    def __init__(
        self,
        model,
        direction: str = "en-tw",
        lookup: Optional[TwiLookup] = None,
        custom_glossary: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.direction = direction.lower()
        self.src_lang = "en" if self.direction.startswith("en") else "tw"
        self.tgt_lang = "tw" if self.direction.startswith("en") else "en"
        self.lookup = lookup or TwiLookup()
        self.custom_glossary = custom_glossary or {}

    def translate(
        self,
        text: str,
        custom_glossary: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        1. Extract source noun phrases.
        2. Generate translation (model preserves source nouns).
        3. Substitute preserved nouns using dictionary.
        """
        if not text or not text.strip():
            return {"translation": "", "raw": "", "substitutions": []}

        # 1. Extract source noun phrases
        source_nouns = extract_noun_phrases(text, lang=self.src_lang)

        # 2. Model generation (preserves source nouns in output)
        raw_translation = self.model.generate(text)

        # 3. Post-translation glossary substitution
        merged_glossary = {**self.custom_glossary, **(custom_glossary or {})}
        final_translation, substitutions = substitute_glossary(
            translation=raw_translation,
            source_terms=source_nouns,
            lookup=self.lookup,
            direction=self.direction,
            custom_glossary=merged_glossary,
        )

        return {
            "translation": final_translation,
            "raw_translation": raw_translation,
            "source_nouns": source_nouns,
            "substitutions": substitutions,
        }
