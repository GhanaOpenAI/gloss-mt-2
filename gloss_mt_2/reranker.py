"""Contextual Lexical Reranker for GLOSS-MT-2."""

import re
from typing import List, Optional


class ContextualReranker:
    """Reranks candidate Twi translations for an English noun phrase in sentence context."""

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
        context: str,
        noun_phrase: str,
        candidates: List[str],
    ) -> str:
        """Given sentence context and candidates, returns the candidate with the highest score."""
        if not candidates:
            return noun_phrase
        if len(candidates) == 1:
            return candidates[0]

        # Ensure noun_phrase is marked in context if not already
        if f"<{noun_phrase}>" not in context:
            pat = re.compile(r"(?<!\w)" + re.escape(noun_phrase) + r"(?!\w)", re.IGNORECASE)
            marked_context = pat.sub(f"<{noun_phrase}>", context, count=1)
        else:
            marked_context = context

        pairs_a = [marked_context] * len(candidates)
        pairs_b = candidates

        enc = self.tok(
            pairs_a,
            pairs_b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**enc).logits.squeeze(-1)

        best_idx = int(torch.argmax(logits).item())
        return candidates[best_idx]
