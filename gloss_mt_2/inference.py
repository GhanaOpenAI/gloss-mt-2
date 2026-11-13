"""Post-translation glossary substitution and inference pipeline for GLOSS-MT-2."""

import re
from typing import Dict, List, Optional, Tuple

from .lookup import TwiLookup, normalize_term

_BRACKETED = re.compile(r"<([^<>\n]{1,120})>")


def substitute_glossary(
    translation: str,
    lookup: Optional[TwiLookup] = None,
    direction: str = "en-tw",
    custom_glossary: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Finds every `<term>` in the model's generated translation:
    - If `term` is found in custom_glossary or lookup dictionary -> replaces with its translation.
    - If `term` is not found (e.g. proper name, unknown entity) -> strips brackets and preserves the term.
    """
    applied = []

    def sub_fn(m):
        term = m.group(1).strip()
        norm = normalize_term(term)
        trans = None

        if custom_glossary:
            for k, v in custom_glossary.items():
                if normalize_term(k) == norm:
                    trans = v
                    break

        if trans is None and lookup is not None:
            trans = lookup.lookup(norm, direction=direction)

        if trans:
            # Preserve capitalization of the source match
            res = trans
            if term.isupper() and len(term) > 1:
                res = res.upper()
            elif term[:1].isupper() and not res[:1].isupper():
                res = res[:1].upper() + res[1:]
            applied.append({"source": term, "target": res, "substituted": True})
            return res
        else:
            # Not in dictionary: leave preserved source term cleanly without brackets
            applied.append({"source": term, "target": term, "substituted": False})
            return term

    final_text = _BRACKETED.sub(sub_fn, translation)
    # Collapse any stray double spaces
    final_text = re.sub(r" +", " ", final_text).strip()
    return final_text, applied


class GlossMT2Translator:
    """End-to-end translator for GLOSS-MT-2 models."""

    def __init__(
        self,
        model,
        direction: str = "en-tw",
        lookup: Optional[TwiLookup] = None,
        custom_glossary: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.direction = direction.lower()
        self.lookup = lookup or TwiLookup()
        self.custom_glossary = custom_glossary or {}

    def translate(
        self,
        text: str,
        custom_glossary: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Translates raw input text (no preprocessing required) and applies post-generation substitution.
        """
        if not text or not text.strip():
            return {"translation": "", "raw_translation": "", "substitutions": []}

        # 1. Model generation (accepts plain, un-preprocessed text; outputs text with <untranslated_nouns>)
        raw_translation = self.model.generate(text)

        # 2. Post-generation glossary substitution
        merged_glossary = {**self.custom_glossary, **(custom_glossary or {})}
        final_translation, substitutions = substitute_glossary(
            translation=raw_translation,
            lookup=self.lookup,
            direction=self.direction,
            custom_glossary=merged_glossary,
        )

        return {
            "translation": final_translation,
            "raw_translation": raw_translation,
            "substitutions": substitutions,
        }
