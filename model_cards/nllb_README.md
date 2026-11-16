---
language:
- en
- tw
license: cc-by-nc-4.0
base_model: facebook/nllb-200-distilled-600M
tags:
- translation
- seq2seq
- machine-translation
- akan
- twi
- asante-twi
- nllb
- nkrane-mt
- african-nlp
pipeline_tag: translation
metrics:
- bleu
- chrf
model-index:
- name: nkrane-mt-nllb-600m-en2tw
  results:
  - task:
      type: translation
      name: English to Asante Twi Translation
    dataset:
      name: nkrane-mt-parallel-corpus-eng-twi
      type: ghanaopenai/nkrane-mt-parallel-corpus-eng-twi
    metrics:
    - type: bleu
      value: 31.57
      name: SacreBLEU (Substituted)
    - type: chrf
      value: 53.34
      name: chrF2 (Substituted)
    - type: bleu
      value: 24.48
      name: SacreBLEU (Raw Preserved)
---

# Nkrane-MT NLLB-200 600M (`nkrane-mt-nllb-600m-en2tw`)

**`ghanaopenai/nkrane-mt-nllb-600m-en2tw`** is an entity-grounded English $\to$ Asante Twi translation model based on Meta's `facebook/nllb-200-distilled-600M`, fine-tuned on the 769,594-pair [Nkrane-MT Parallel Corpus](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi) using dual-sided bracket entity preservation.

Part of the [**Nkrane-MT**](https://github.com/GhanaOpenAI/nkrane-mt) pipeline developed by [GhanaOpenAI](https://github.com/GhanaOpenAI).

---

## 🎯 Benchmark Performance

Evaluated on held-out test splits with live SacreBLEU, chrF2, and post-translation dictionary substitution:

| Metric | Substituted (Final Output) | Raw (Bracketed Generation) |
| :--- | :---: | :---: |
| **SacreBLEU** | **31.57** | 24.48 |
| **chrF2** | **53.34** | 49.66 |
| **Evaluation Loss** | **0.6976** | - |

*Note: NLLB-200 600M outperforms the OPUS-MT model by **+1.85 SacreBLEU** and **+3.00 chrF2**, with validation loss dropping from 1.551 to 0.6976.*

---

## 🚀 Usage

### 1. Direct Translation (Pure Transformers)
To translate plain English directly into Asante Twi:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ghanaopenai/nkrane-mt-nllb-600m-en2tw")
model = AutoModelForSeq2SeqLM.from_pretrained("ghanaopenai/nkrane-mt-nllb-600m-en2tw")

tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "twi_Latn"
forced_bos = tokenizer.convert_tokens_to_ids("twi_Latn")

text = "The president of Ghana visited the military hospital in Accra yesterday."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=128)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

### 2. Entity-Grounded Pipeline (`nkrane_mt`)
To automatically detect and preserve entities, resolving them through an offline glossary and contextual reranker:

```bash
pip install git+https://github.com/GhanaOpenAI/nkrane-mt.git
```

```python
from nkrane_mt.inference import NkraneMTTranslator

translator = NkraneMTTranslator(
    model="ghanaopenai/nkrane-mt-nllb-600m-en2tw",
    marker_model_name="ghanaopenai/nkrane-mt-marker-en",
    use_default_dictionary=True,
)

result = translator.translate("The president of Ghana visited the military hospital in Accra yesterday.")
print(result["translation"])
```

You can also pass custom domain terminology:
```python
result = translator.translate(
    "The doctor gave medicine to the patient.",
    custom_glossary={"patient": "ɔyarefoɔ"}
)
```

---

## 🏗️ Architecture & Training

- **Base Architecture:** NLLB-200 Distilled 600M (`facebook/nllb-200-distilled-600M`)
- **Dataset:** 763,594 training pairs ([`ghanaopenai/nkrane-mt-parallel-corpus-eng-twi`](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi))
- **Precision:** `bfloat16` + `TF32` on NVIDIA H200
- **Batch Size:** 16 per device $\times$ 2 gradient accumulation steps (effective batch size 32)
- **Learning Rate:** 1e-4 with AdamW and linear warmup
- **Selected Checkpoint:** Step 110,000 (Epoch 4.61, top composite score: 44.90)

---

## 📄 Citation

```bibtex
@misc{nkrane-mt-2026,
  author = {GhanaOpenAI},
  title = {Nkrane-MT: Entity-Grounded English-Twi Translation Pipeline},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/ghanaopenai/nkrane-mt-nllb-600m-en2tw}}
}
```
