---
language:
- en
- tw
license: apache-2.0
base_model: google/umt5-base
tags:
- translation
- seq2seq
- machine-translation
- akan
- twi
- asante-twi
- umt5
- t5
- nkrane-mt
- african-nlp
pipeline_tag: translation
metrics:
- bleu
- chrf
model-index:
- name: nkrane-mt-umt5-base-en2tw
  results:
  - task:
      type: translation
      name: English to Asante Twi Translation
    dataset:
      name: nkrane-mt-parallel-corpus-eng-twi
      type: ghanaopenai/nkrane-mt-parallel-corpus-eng-twi
    metrics:
    - type: bleu
      value: 22.01
      name: SacreBLEU (Substituted)
    - type: chrf
      value: 41.75
      name: chrF2 (Substituted)
    - type: bleu
      value: 17.14
      name: SacreBLEU (Raw Preserved)
---

# Nkrane-MT uMT5-Base (`nkrane-mt-umt5-base-en2tw`)

**`ghanaopenai/nkrane-mt-umt5-base-en2tw`** is an entity-grounded English $\to$ Asante Twi translation model based on Google's `google/umt5-base` (Uni-max Multilingual T5, 580M parameters), fine-tuned on the 769,594-pair [Nkrane-MT Parallel Corpus](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi) using dual-sided bracket entity preservation.

Part of the [**Nkrane-MT**](https://github.com/GhanaOpenAI/nkrane-mt) pipeline developed by [GhanaOpenAI](https://github.com/GhanaOpenAI).

---

## 🎯 Benchmark Performance

Evaluated on held-out test splits with live SacreBLEU, chrF2, and post-translation dictionary substitution:

| Metric | Substituted (Final Output) | Raw (Bracketed Generation) |
| :--- | :---: | :---: |
| **SacreBLEU** | **22.01** | 17.14 |
| **chrF2** | **41.75** | 38.56 |
| **Evaluation Loss** | **1.327** | - |

---

## 🚀 Usage

### 1. Direct Translation (Pure Transformers)
To translate plain English directly into Asante Twi:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ghanaopenai/nkrane-mt-umt5-base-en2tw")
model = AutoModelForSeq2SeqLM.from_pretrained("ghanaopenai/nkrane-mt-umt5-base-en2tw")

text = "The president of Ghana visited the military hospital in Accra yesterday."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
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
    model="ghanaopenai/nkrane-mt-umt5-base-en2tw",
    marker_model_name="ghanaopenai/nkrane-mt-marker-en",
    use_default_dictionary=True,
)

result = translator.translate("The president of Ghana visited the military hospital in Accra yesterday.")
print(result["translation"])
```

---

## 🏗️ Architecture & Training

- **Base Architecture:** Google uMT5-Base (`google/umt5-base`)
- **Parameters:** 580 Million
- **Dataset:** 763,594 training pairs ([`ghanaopenai/nkrane-mt-parallel-corpus-eng-twi`](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi))
- **Precision:** `bfloat16` + `TF32` on NVIDIA H200
- **Batch Size:** 16 per device $\times$ 2 gradient accumulation steps (effective batch size 32)
- **Learning Rate:** 2e-4 with AdamW and linear warmup
- **Training Steps:** 181,000 steps (~7.58 epochs) to convergence

---

## 📄 Citation

```bibtex
@misc{nkrane-mt-2026,
  author = {GhanaOpenAI},
  title = {Nkrane-MT: Entity-Grounded English-Twi Translation Pipeline},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/ghanaopenai/nkrane-mt-umt5-base-en2tw}}
}
```
