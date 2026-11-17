"""Post-translation glossary substitution and inference pipeline for GLOSS-MT-2."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .lookup import TwiLookup, normalize_term
from .marker import NounPhraseMarker
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
    """End-to-end translator for Nkrane-MT (formerly GLOSS-MT-2) models with optional contextual reranking.

    Accepts a Hugging Face model identifier/path (str) or a model object with a `.generate()` method.
    By default, the heavy offline dictionary is NOT loaded unless explicitly requested via
    `use_default_dictionary=True` or `lookup=TwiLookup()`. Custom glossaries can be passed
    as lists, dicts, or files.

    At inference time, `NounPhraseMarker` automatically detects and brackets targeted noun
    phrases on the input side before feeding to the seq2seq generator.
    """

    def __init__(
        self,
        model: Any,
        direction: str = "en-tw",
        lookup: Optional[TwiLookup] = None,
        reranker: Optional[ContextualReranker] = None,
        custom_glossary: Optional[Union[Dict, List]] = None,
        use_default_dictionary: bool = False,
        marker: Optional[NounPhraseMarker] = None,
        marker_model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.direction = direction.lower()
        self.reranker = reranker

        # Setup marker (lazy-loaded so direct translation starts in milliseconds)
        self._marker = marker
        self._marker_model_name = marker_model_name

        # Setup model / tokenizer
        self._is_hf_model = False
        self.tokenizer = None
        self.raw_model = None

        if isinstance(model, str):
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

            if "umt5" in model.lower():
                from transformers import UMT5Config, UMT5ForConditionalGeneration

                cfg = UMT5Config.from_pretrained(model)
                cfg.tie_word_embeddings = False
                cfg.model_type = "umt5"
                self.raw_model = UMT5ForConditionalGeneration.from_pretrained(model, config=cfg, torch_dtype=dtype).to(self.device)
            else:
                self.raw_model = AutoModelForSeq2SeqLM.from_pretrained(model, dtype=dtype).to(self.device)

            self.raw_model.eval()
            self._is_hf_model = True
        else:
            self.raw_model = model
            self.tokenizer = getattr(model, "tokenizer", None)
            self._is_hf_model = False

        self.model = model

        if lookup is not None:
            self.lookup = lookup
        elif use_default_dictionary:
            self.lookup = TwiLookup()
        else:
            self.lookup = None

        self.custom_glossary = normalize_glossary_input(custom_glossary)

    @property
    def marker(self) -> NounPhraseMarker:
        if self._marker is None:
            if self._marker_model_name:
                self._marker = NounPhraseMarker(method="model", model_name_or_path=self._marker_model_name)
            else:
                self._marker = NounPhraseMarker(method="rules")
        return self._marker

    def translate(
        self,
        text: str,
        custom_glossary: Optional[Union[Dict, List]] = None,
        auto_mark: bool = True,
        skip_substitution: bool = False,
        direct: bool = False,
        max_src_length: int = 512,
        max_new_tokens: int = 512,
    ) -> Dict:
        """
        Translates input text and applies optional smart contextual substitution.

        Args:
            text: Input text to translate (supports single sentences or paragraphs up to 512 tokens).
            custom_glossary: Optional dictionary or list of terms to enforce.
            auto_mark: Whether to automatically detect and bracket entities in input.
            skip_substitution: If True, skips dictionary substitution and returns clean model output.
            direct: Shortcut for direct translation (sets auto_mark=False and skip_substitution=True).
            max_src_length: Max input tokens (models trained up to 512 tokens).
            max_new_tokens: Max generated tokens (models trained up to 512 tokens).
        """
        if direct:
            auto_mark = False
            skip_substitution = True

        if not text or not text.strip():
            return {"translation": "", "raw_translation": "", "marked_input": "", "substitutions": []}

        merged_glossary = dict(self.custom_glossary)
        if custom_glossary:
            merged_glossary.update(normalize_glossary_input(custom_glossary))

        # Mark noun phrases on input side if auto_mark is True and text is not already bracketed
        if auto_mark and ("<" not in text or ">" not in text):
            custom_keys = [k for k in merged_glossary.keys() if isinstance(k, str)]
            marked_input, _ = self.marker.mark(text, custom_terms=custom_keys)
        else:
            marked_input = text

        if self._is_hf_model:
            import torch

            inputs = self.tokenizer(marked_input, return_tensors="pt", truncation=True, max_length=max_src_length).to(self.device)
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1}
            if hasattr(self.tokenizer, "tgt_lang") and self.tokenizer.tgt_lang:
                gen_kwargs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)
            with torch.inference_mode():
                outputs = self.raw_model.generate(**inputs, **gen_kwargs)
            raw_translation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        elif hasattr(self.raw_model, "generate"):
            raw_translation = self.raw_model.generate(marked_input)
        else:
            raw_translation = str(self.raw_model(marked_input))

        if skip_substitution:
            clean_text = re.sub(r"<([^<>\n]+)>", r"\1", raw_translation)
            clean_text = re.sub(r" +", " ", clean_text).strip()
            return {
                "translation": clean_text,
                "raw_translation": raw_translation,
                "marked_input": marked_input,
                "substitutions": [],
            }

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
            "marked_input": marked_input,
            "substitutions": substitutions,
        }

    def translate_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        max_length: int = 512,
        max_src_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        auto_mark: bool = False,
        skip_substitution: bool = True,
        custom_glossary: Optional[Union[Dict, List]] = None,
        return_details: bool = False,
        show_progress: bool = False,
    ) -> Union[List[str], List[Dict]]:
        """
        High-throughput batched translation optimized for large datasets and full GPU saturation.

        By default, runs in direct translation mode (auto_mark=False, skip_substitution=True)
        to bypass CPU dictionary bottlenecks and achieve maximum tokens/sec on ANY GPU.

        Supports full multi-sentence paragraphs up to 512 tokens (the models' trained capacity).
        Uses dynamic batch padding so short batches remain fast while long batches utilize the full window.

        Args:
            texts: List of strings (sentences or paragraphs) to translate.
            batch_size: Number of texts per GPU batch (e.g. 32-128 depending on GPU VRAM).
            max_length: Default sequence length ceiling (512 tokens).
            max_src_length: Optional explicit input length limit (defaults to max_length=512).
            max_new_tokens: Optional explicit output length limit (defaults to max_length=512).
            auto_mark: Whether to run noun phrase marking on input texts.
            skip_substitution: If True, returns clean direct translation without dictionary lookup.
            custom_glossary: Optional dictionary of terminology to substitute.
            return_details: If True, returns List[Dict] with metadata. If False (default), returns List[str].
            show_progress: If True, shows tqdm progress bar.

        Returns:
            List of translated strings, or list of result dicts if return_details=True.
        """
        if not texts:
            return []

        src_limit = max_src_length or max_length
        tgt_limit = max_new_tokens or max_length

        merged_glossary = dict(self.custom_glossary)
        if custom_glossary:
            merged_glossary.update(normalize_glossary_input(custom_glossary))

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc="Translating batches")
            except ImportError:
                pass

        results: List[Any] = []

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            if not batch_texts:
                continue

            # Marking phase (optional)
            if auto_mark:
                custom_keys = [k for k in merged_glossary.keys() if isinstance(k, str)]
                marked_batch = [
                    self.marker.mark(t, custom_terms=custom_keys)[0] if ("<" not in t or ">" not in t) else t
                    for t in batch_texts
                ]
            else:
                marked_batch = batch_texts

            # Generation phase
            if self._is_hf_model:
                import torch

                inputs = self.tokenizer(
                    marked_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=src_limit,
                ).to(self.device)

                gen_kwargs = {
                    "max_new_tokens": tgt_limit,
                    "do_sample": False,
                    "num_beams": 1,
                }
                if hasattr(self.tokenizer, "tgt_lang") and self.tokenizer.tgt_lang:
                    gen_kwargs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)

                with torch.inference_mode():
                    outputs = self.raw_model.generate(**inputs, **gen_kwargs)

                raw_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                raw_outputs = [r.strip() for r in raw_outputs]
            elif hasattr(self.raw_model, "generate"):
                raw_outputs = [self.raw_model.generate(m) for m in marked_batch]
            else:
                raw_outputs = [str(self.raw_model(m)) for m in marked_batch]

            # Output processing phase
            for inp, raw in zip(marked_batch, raw_outputs):
                if skip_substitution:
                    clean_text = re.sub(r"<([^<>\n]+)>", r"\1", raw)
                    clean_text = re.sub(r" +", " ", clean_text).strip()
                    if return_details:
                        results.append({
                            "translation": clean_text,
                            "raw_translation": raw,
                            "marked_input": inp,
                            "substitutions": [],
                        })
                    else:
                        results.append(clean_text)
                else:
                    final_text, subs = substitute_glossary(
                        translation=raw,
                        lookup=self.lookup,
                        direction=self.direction,
                        custom_glossary=merged_glossary,
                        reranker=self.reranker,
                        source_context=raw,
                    )
                    if return_details:
                        results.append({
                            "translation": final_text,
                            "raw_translation": raw,
                            "marked_input": inp,
                            "substitutions": subs,
                        })
                    else:
                        results.append(final_text)

        return results

    def translate_stream(
        self,
        texts_iter: Any,
        batch_size: int = 64,
        max_length: int = 512,
        max_src_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        auto_mark: bool = False,
        skip_substitution: bool = True,
        custom_glossary: Optional[Union[Dict, List]] = None,
    ):
        """
        Streaming batched generator for multi-gigabyte datasets (never loads full corpus into RAM).
        Yields translated strings batch by batch.
        """
        chunk = []
        for text in texts_iter:
            chunk.append(text)
            if len(chunk) >= batch_size:
                yield from self.translate_batch(
                    chunk,
                    batch_size=batch_size,
                    max_length=max_length,
                    max_src_length=max_src_length,
                    max_new_tokens=max_new_tokens,
                    auto_mark=auto_mark,
                    skip_substitution=skip_substitution,
                    custom_glossary=custom_glossary,
                    return_details=False,
                )
                chunk = []
        if chunk:
            yield from self.translate_batch(
                chunk,
                batch_size=batch_size,
                max_length=max_length,
                max_src_length=max_src_length,
                max_new_tokens=max_new_tokens,
                auto_mark=auto_mark,
                skip_substitution=skip_substitution,
                custom_glossary=custom_glossary,
                return_details=False,
            )


# Alias for backwards and modern compatibility
NkraneMTTranslator = GlossMT2Translator
