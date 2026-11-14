"""Seq2Seq model wrappers for NLLB and OPUS-MT (Marian)."""

from typing import List, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


NLLB_LANGS = {
    "en": "eng_Latn",
    "tw": "twi_Latn",
}


class Seq2SeqModelWrapper:
    def __init__(
        self,
        model_name_or_path: str,
        direction: str = "en-tw",
        device: Optional[str] = None,
        max_length: int = 1024,
        num_beams: int = 4,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.direction = direction.lower()
        self.max_length = max_length
        self.num_beams = num_beams

        dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32

        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=dtype).to(self.device).eval()

        # Check if NLLB
        self.is_nllb = "nllb" in model_name_or_path.lower()
        if self.is_nllb:
            src_key = "en" if self.direction.startswith("en") else "tw"
            tgt_key = "tw" if self.direction.startswith("en") else "en"
            self.tok.src_lang = NLLB_LANGS[src_key]
            self.tok.tgt_lang = NLLB_LANGS[tgt_key]
            self.forced_bos_token_id = self.tok.convert_tokens_to_ids(NLLB_LANGS[tgt_key])
        else:
            self.forced_bos_token_id = None

    def generate(self, text: str) -> str:
        return self.generate_batch([text])[0]

    def generate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        inputs = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        gen_kwargs = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "do_sample": False,
        }
        if self.forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = self.forced_bos_token_id

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tok.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in decoded]
