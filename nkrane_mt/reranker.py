"""Contextual Lexical Reranker for GLOSS-MT-2."""

import re
from typing import List, Optional


class ContextualReranker:
    """Reranks candidate Twi translations for an English noun phrase embedded in a Twi sentence."""

    def __init__(
        self,
        model_name_or_path: str = "Davlan/afro-xlmr-base",
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device).eval()

    def rerank(
        self,
        twi_context: str,
        noun_phrase: str,
        candidates: List[str],
    ) -> str:
        """
        Given the generated Twi sentence containing <noun_phrase> and candidate Twi translations:
        returns the candidate with the highest contextual logit.
        """
        if not candidates:
            return noun_phrase
        if len(candidates) == 1:
            return candidates[0]

        pairs_a = [twi_context] * len(candidates)
        pairs_b = candidates

        enc = self.tok(
            pairs_a,
            pairs_b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with self.torch.inference_mode():
            logits = self.model(**enc).logits.squeeze(-1)

        best_idx = int(self.torch.argmax(logits).item())
        return candidates[best_idx]
