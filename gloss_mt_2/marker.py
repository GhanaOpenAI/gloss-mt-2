"""Noun Phrase Marker for GLOSS-MT-2.

Marks targeted noun phrases and entities in English inputs with angle brackets `<...>`
before passing to seq2seq translation models.
Supports both fast grammatical rules and fine-tuned token classification models.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Union

from .extractor import extract_selective_noun_phrases, get_nlp


def mark_text_with_phrases(text: str, phrases: List[str]) -> str:
    """
    Safely wraps each occurrence of `phrases` with angle brackets `<phrase>`,
    avoiding double-bracketing, partial word corruption, or nesting.
    """
    if not text or not phrases:
        return text

    sorted_phrases = sorted(
        set(p.strip() for p in phrases if p and p.strip()),
        key=lambda p: (len(p.split()), len(p)),
        reverse=True,
    )

    bracket_pool = []

    def save_bracket(m):
        idx = len(bracket_pool)
        bracket_pool.append(m.group(0))
        return f"___BRACKET_{idx}___"

    # Mask pre-existing brackets in user prompt
    working = re.sub(r"<[^<>\n]+>", save_bracket, text)

    for p_clean in sorted_phrases:
        flags = re.IGNORECASE if p_clean.islower() else 0
        pattern = re.compile(rf"\b({re.escape(p_clean)})\b", flags)

        def repl(m):
            idx = len(bracket_pool)
            bracket_pool.append(f"<{m.group(1)}>")
            return f"___BRACKET_{idx}___"

        working = pattern.sub(repl, working)

    # Restore all brackets
    for idx, orig in enumerate(bracket_pool):
        working = working.replace(f"___BRACKET_{idx}___", orig)

    return working


class NounPhraseMarker:
    """
    Identifies and brackets noun phrases in input text at inference time:
    - `method="rules"`: Uses grammatical dependency parsing and WordNet abstract filter.
    - `method="model"`: Uses a trained sequence classification / NER model.
    """

    def __init__(
        self,
        method: str = "rules",
        model_name_or_path: Optional[str] = None,
        nlp=None,
        device: Optional[str] = None,
    ):
        self.method = method
        self.nlp = nlp or get_nlp()
        self.model = None
        self.tokenizer = None
        self.device = device

        if method == "model" and model_name_or_path:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path).to(self.device).eval()

    def mark(
        self,
        text: str,
        custom_terms: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Marks targeted phrases in input English sentence.
        Returns:
            (marked_text, list_of_marked_phrases)
        """
        if not text or not text.strip():
            return text, []

        all_terms = []
        if self.method == "rules" or not self.model:
            extracted = extract_selective_noun_phrases(text, nlp=self.nlp)
            all_terms = list(extracted)
        else:
            # Model-based inference
            import torch
            enc = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            preds = torch.argmax(logits, dim=-1)[0].tolist()

            spans = []
            curr_start = None
            curr_end = None
            for (start, end), label_id in zip(offsets, preds):
                if start == end:
                    continue
                label = self.model.config.id2label.get(label_id, "O")
                if label == "B-PRESERVE":
                    if curr_start is not None:
                        spans.append((curr_start, curr_end))
                    curr_start, curr_end = start, end
                elif label == "I-PRESERVE":
                    if curr_start is not None:
                        curr_end = end
                else:
                    if curr_start is not None:
                        spans.append((curr_start, curr_end))
                        curr_start, curr_end = None, None
            if curr_start is not None:
                spans.append((curr_start, curr_end))

            for s, e in spans:
                phrase = text[s:e].strip()
                if phrase and phrase not in all_terms:
                    all_terms.append(phrase)

        if custom_terms:
            for ct in custom_terms:
                if ct and ct.strip() and ct.strip() not in all_terms:
                    all_terms.append(ct.strip())

        marked_text = mark_text_with_phrases(text, all_terms)
        found = re.findall(r"<([^<>\n]+)>", marked_text)
        return marked_text, found
