"""Bidirectional offline English <-> Twi dictionary lookup with custom glossary support."""

import gzip
import os
import re
from typing import Dict, List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "twi_dictionary_single.tsv.gz")

_WS = re.compile(r"\s+")


def normalize_term(term: str) -> str:
    """Lowercase, trim, and collapse whitespace."""
    return _WS.sub(" ", term).strip().lower()


class TwiLookup:
    """Offline bidirectional dictionary supporting both en->tw and tw->en lookups."""

    def __init__(self, path: str = DEFAULT_PATH, custom_dict: Optional[Dict[str, str]] = None):
        self.en2tw: Dict[str, str] = {}
        self.tw2en: Dict[str, str] = {}

        if os.path.exists(path):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        en, tw = parts[0].strip(), parts[1].strip()
                        en_norm = normalize_term(en)
                        tw_norm = normalize_term(tw)
                        if en_norm and tw_norm:
                            self.en2tw[en_norm] = tw
                            if tw_norm not in self.tw2en:
                                self.tw2en[tw_norm] = en

        # Overlay custom dictionary if provided
        if custom_dict:
            for k, v in custom_dict.items():
                k_norm, v_norm = normalize_term(k), normalize_term(v)
                self.en2tw[k_norm] = v
                self.tw2en[v_norm] = k

    def lookup(self, term: str, direction: str = "en-tw") -> Optional[str]:
        norm = normalize_term(term)
        if direction == "en-tw":
            return self.en2tw.get(norm)
        elif direction == "tw-en":
            return self.tw2en.get(norm)
        return None

    def lookup_many(self, terms: List[str], direction: str = "en-tw") -> Dict[str, str]:
        results = {}
        for t in terms:
            trans = self.lookup(t, direction=direction)
            if trans:
                results[t] = trans
        return results

    def __len__(self):
        return len(self.en2tw)
