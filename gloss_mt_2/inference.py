"""Post-translation glossary substitution and inference pipeline for GLOSS-MT-2."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .lookup import TwiLookup, normalize_term
from .reranker import ContextualReranker

_BRACKETED = re.compile(r"<([^<>\n]{1,120})>")


def normalize_glossary_input(glossary_input: Any) -> Dict[str, Union[str, List[str]]]:
    """
    Normalizes custom glossary from various flexible formats:
    - Dict: {"trade": ["dwadie", "aguadi"], "court": "asɛnnibea"}
    - List of tuples/pairs: [("trade", ["dwadie", "aguadi"]), ("court", "asɛnnibea")]
    - List of dicts: [{"source": "trade", "candidates": ["dwadie", "aguadi"]}]
    - List of strings: ["trade\\tdwadie\\taguadi", "court\\tasɛnnibea"]
    - File path string: path to .tsv, .csv, or .json file
    """
    if not glossary_input:
        return {}

    # File path
    if isinstance(glossary_input, str) and os.path.isfile(glossary_input):
        result = {}
        if glossary_input.endswith(".json"):
            with open(glossary_input, "r", encoding="utf-8") as f:
                data = json.load(f)
                return normalize_glossary_input(data)
        else:
            with open(glossary_input, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t") if "\t" in line else line.split(",")
                    if len(parts) >= 2:
                        k = parts[0].strip()
                        cands = [c.strip() for c in parts[1:] if c.strip()]
                        result[k] = cands if len(cands) > 1 else (cands[0] if cands else "")
            return result

    # Dict
    if isinstance(glossary_input, dict):
        result = {}
        for k, v in glossary_input.items():
            if isinstance(v, (list, tuple, set)):
                cands = [str(c).strip() for c in v if str(c).strip()]
                result[str(k).strip()] = cands if len(cands) > 1 else (cands[0] if cands else "")
            else:
                result[str(k).strip()] = str(v).strip()
        return result

    # List or tuple
    if isinstance(glossary_input, (list, tuple)):
        result = {}
        for item in glossary_input:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                k = str(item[0]).strip()
                v = item[1]
                if isinstance(v, (list, tuple, set)):
                    cands = [str(c).strip() for c in v if str(c).strip()]
                    result[k] = cands if len(cands) > 1 else (cands[0] if cands else "")
                else:
                    result[k] = str(v).strip()
            elif isinstance(item, dict):
                src = item.get("source") or item.get("en") or item.get("term")
                tgt = item.get("target") or item.get("tw") or item.get("candidates") or item.get("translation")
                if src and tgt:
                    if isinstance(tgt, (list, tuple, set)):
                        cands = [str(c).strip() for c in tgt if str(c).strip()]
                        result[str(src).strip()] = cands if len(cands) > 1 else (cands[0] if cands else "")
                    else:
                        result[str(src).strip()] = str(tgt).strip()
            elif isinstance(item, str):
                line = item.strip()
                parts = line.split("\t") if "\t" in line else line.split(",")
                if len(parts) >= 2:
                    k = parts[0].strip()
                    cands = [c.strip() for c in parts[1:] if c.strip()]
                    result[k] = cands if len(cands) > 1 else (cands[0] if cands else "")
        return result

    return {}


def substitute_glossary(
    translation: str,
    lookup: Optional[TwiLookup] = None,
    direction: str = "en-tw",
    custom_glossary: Optional[Union[Dict, List]] = None,
    reranker: Optional[ContextualReranker] = None,
    source_context: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Finds every `<term>` in the model's generated translation:
    - If `custom_glossary` is provided (dict, list of pairs, etc.), matching terms are substituted.
      If a custom term has multiple candidates, the reranker evaluates them in context.
    - If `lookup` is provided, looks up candidates in the dictionary.
    - If `term` is not found (e.g. unknown entity, proper noun) -> strips brackets and preserves the term cleanly.
    """
    applied = []
    norm_glossary = normalize_glossary_input(custom_glossary)

    def sub_fn(m):
        term = m.group(1).strip()
        norm = normalize_term(term)
        trans = None

        if norm_glossary:
            for k, v in norm_glossary.items():
                if normalize_term(k) == norm:
                    if isinstance(v, list) and len(v) > 1:
                        if reranker is not None and source_context:
                            trans = reranker.rerank(
                                twi_context=source_context,
                                noun_phrase=term,
                                candidates=v,
                            )
                        else:
                            trans = v[0]
                    elif isinstance(v, list) and len(v) == 1:
                        trans = v[0]
                    elif isinstance(v, str):
                        trans = v
                    break

        if trans is None and lookup is not None:
            candidates = lookup.get_candidates(norm, direction=direction) if hasattr(lookup, "get_candidates") else getattr(lookup, "candidates", {}).get(norm, [])
            if candidates and len(candidates) > 1 and reranker is not None and source_context:
                trans = reranker.rerank(
                    twi_context=source_context,
                    noun_phrase=term,
                    candidates=candidates,
                )
            else:
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
            # Not in dictionary/glossary: leave preserved source term cleanly without brackets
            applied.append({"source": term, "target": term, "substituted": False})
            return term

    final_text = _BRACKETED.sub(sub_fn, translation)
    final_text = re.sub(r" +", " ", final_text).strip()
    return final_text, applied


class GlossMT2Translator:
    """End-to-end translator for GLOSS-MT-2 models with optional contextual reranking.

    By default, the heavy offline dictionary is NOT loaded unless explicitly requested via
    `use_default_dictionary=True` or `lookup=TwiLookup()`. Custom glossaries can be passed
    as lists, dicts, or files.
    """

    def __init__(
        self,
        model,
        direction: str = "en-tw",
        lookup: Optional[TwiLookup] = None,
        reranker: Optional[ContextualReranker] = None,
        custom_glossary: Optional[Union[Dict, List]] = None,
        use_default_dictionary: bool = False,
    ):
        self.model = model
        self.direction = direction.lower()
        self.reranker = reranker

        if lookup is not None:
            self.lookup = lookup
        elif use_default_dictionary:
            self.lookup = TwiLookup()
        else:
            self.lookup = None

        self.custom_glossary = normalize_glossary_input(custom_glossary)

    def translate(
        self,
        text: str,
        custom_glossary: Optional[Union[Dict, List]] = None,
    ) -> Dict:
        """
        Translates raw input text (no preprocessing required) and applies smart contextual substitution.
        """
        if not text or not text.strip():
            return {"translation": "", "raw_translation": "", "substitutions": []}

        raw_translation = self.model.generate(text)

        merged_glossary = dict(self.custom_glossary)
        if custom_glossary:
            merged_glossary.update(normalize_glossary_input(custom_glossary))

        final_translation, substitutions = substitute_glossary(
            translation=raw_translation,
            lookup=self.lookup,
            direction=self.direction,
            custom_glossary=merged_glossary,
            reranker=self.reranker,
            source_context=raw_translation,
        )

        return {
            "translation": final_translation,
            "raw_translation": raw_translation,
            "substitutions": substitutions,
        }
