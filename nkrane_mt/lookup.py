"""Bidirectional offline English <-> Twi dictionary lookup with custom glossary support."""

import gzip
import os
import re
from typing import Dict, List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "twi_dictionary.tsv.gz")
FALLBACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "twi_dictionary_single.tsv.gz")

_WS = re.compile(r"\s+")


def normalize_term(term: str) -> str:
    """Lowercase, trim, and collapse whitespace."""
    return _WS.sub(" ", term).strip().lower()


class TwiLookup:
    """Offline bidirectional dictionary supporting both en->tw and tw->en lookups, with candidate lists for reranking."""

    def __init__(self, path: Optional[str] = None, custom_dict: Optional[Dict] = None):
        self.en2tw: Dict[str, str] = {}
        self.tw2en: Dict[str, str] = {}
        self.candidates: Dict[str, List[str]] = {}

        dict_path = path or (DEFAULT_PATH if os.path.exists(DEFAULT_PATH) else FALLBACK_PATH)

        if os.path.exists(dict_path):
            with gzip.open(dict_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2:
                        en = parts[0].strip()
                        en_norm = normalize_term(en)
                        tw_list = [c.strip() for c in parts[1:] if c.strip()]
                        if en_norm and tw_list:
                            self.en2tw[en_norm] = tw_list[0]
                            self.candidates[en_norm] = tw_list
                            primary_tw = tw_list[0]
                            primary_tw_norm = normalize_term(primary_tw)
                            if primary_tw_norm not in self.tw2en:
                                self.tw2en[primary_tw_norm] = en

        # Overlay custom dictionary if provided
        if custom_dict:
            for k, v in custom_dict.items():
                k_norm = normalize_term(k)
                if isinstance(v, list):
                    clean_list = [c.strip() for c in v if c and str(c).strip()]
                    if clean_list:
                        self.candidates[k_norm] = clean_list
                        self.en2tw[k_norm] = clean_list[0]
                        for c in clean_list:
                            c_norm = normalize_term(c)
                            if c_norm not in self.tw2en:
                                self.tw2en[c_norm] = k
                else:
                    v_str = str(v).strip()
                    v_norm = normalize_term(v_str)
                    self.en2tw[k_norm] = v_str
                    self.candidates[k_norm] = [v_str]
                    self.tw2en[v_norm] = k

    def get_candidates(self, term: str, direction: str = "en-tw") -> List[str]:
        """Returns all candidate translations for contextual reranking."""
        norm = normalize_term(term)
        if direction == "en-tw":
            if norm in self.candidates:
                return self.candidates[norm]
            res = self.en2tw.get(norm)
            return [res] if res else []
        elif direction == "tw-en":
            res = self.tw2en.get(norm)
            return [res] if res else []
        return []

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
