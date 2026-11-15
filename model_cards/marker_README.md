---
language:
- en
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- token-classification
- sequence-tagging
- ner
- entity-recognition
- noun-phrase-tagger
- nkrane-mt
- machine-translation
- african-nlp
pipeline_tag: token-classification
metrics:
- f1
- accuracy
- precision
- recall
model-index:
- name: nkrane-mt-marker-en
  results:
  - task:
      type: token-classification
      name: Noun Phrase & Entity Detection for Nkrane-MT
    dataset:
      name: nkrane-mt-parallel-corpus-eng-twi
      type: ghanaopenai/nkrane-mt-parallel-corpus-eng-twi
    metrics:
    - type: f1
      value: 0.8139
    - type: accuracy
      value: 0.9470
    - type: precision
      value: 0.8085
    - type: recall
      value: 0.8193
---

# Nkrane-MT Inference Noun Phrase Marker (`nkrane-mt-marker-en`)

**`ghanaopenai/nkrane-mt-marker-en`** is a fine-tuned token classification model for **detecting and bracketing concrete noun phrases and named entities in English text** at inference time.

It is a component of the [**Nkrane-MT**](https://github.com/GhanaOpenAI/nkrane-mt) translation pipeline developed by [GhanaOpenAI](https://github.com/GhanaOpenAI/nkrane-mt).

---

## 💡 Role in Nkrane-MT

In Nkrane-MT, translation models are trained with dual-sided bracketed entity preservation:
- Concrete noun phrases, organizations, compound titles, and demographic groups are marked with angle brackets `<...>` on the **English input** (e.g. `The <president of Ghana> visited the <regional hospital>.`).
- The seq2seq translation model directly preserves these bracketed terms on the **Asante Twi target** (e.g. `<president of Ghana> no kɔsra <regional hospital> no.`).
- Abstract concepts (*information, development, peace, justice, patience*) are translated freely and naturally into Twi.

**`nkrane-mt-marker-en`** automates the input-side marking in $< 5\text{ ms}$, taking raw English text and placing `<...>` around targeted concrete entities so users can provide natural English without manual tagging.

---

## 📊 Evaluation Metrics

Trained across all **763,594 sentence pairs** of the [`nkrane-mt-parallel-corpus-eng-twi`](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi) dataset:

| Metric | Score |
|---|---|
| **F1 Score** | **`0.8139`** |
| **Token Accuracy** | **`94.70%`** |
| **Precision** | **`80.85%`** |
| **Recall** | **`81.93%`** |
| **Inference Latency** | **`< 5 ms`** |

---

## 🚀 Quickstart & Usage

### 1. Python Inference with `nkrane_mt`

```python
from nkrane_mt.marker import NounPhraseMarker

# Load the trained marker model
marker = NounPhraseMarker(method="model", model_name_or_path="ghanaopenai/nkrane-mt-marker-en")

raw_text = "The president of Ghana visited the Ministry of Finance and the regional hospital in May 2026."
marked_text, phrases = marker.mark(raw_text)

print("Raw English:    ", raw_text)
print("Marked English: ", marked_text)
print("Detected Phrases:", phrases)

# Output:
# Raw English:     The president of Ghana visited the Ministry of Finance and the regional hospital in May 2026.
# Marked English:  The <president of Ghana> visited the <Ministry of Finance> and the <regional hospital> in <May 2026>.
# Detected Phrases: ['president of Ghana', 'Ministry of Finance', 'regional hospital', 'May 2026']
```

---

## 🏋️ Training Setup

- **Base Model:** [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) (66M parameters).
- **Dataset:** 763,594 pairs from [`ghanaopenai/nkrane-mt-parallel-corpus-eng-twi`](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi).
- **Labels:** `O`, `B-PRESERVE`, `I-PRESERVE`.
- **Optimizer:** AdamW (`lr=3e-5`, cosine schedule, warmup steps = 500).
- **Hardware:** Single NVIDIA H200 (140 GB VRAM), bfloat16.

---

## 📖 Citation & Repository

```bibtex
@misc{nkrane_mt_marker_2026,
  author = {Owusu, S. and GhanaOpenAI Contributors},
  title = {Nkrane-MT: Lightweight Token Classifier for Concrete Noun Phrase & Entity Marking},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/ghanaopenai/nkrane-mt-marker-en}}
}
```

- **GitHub Repository:** [`GhanaOpenAI/nkrane-mt`](https://github.com/GhanaOpenAI/nkrane-mt)
- **Parallel Corpus:** [`ghanaopenai/nkrane-mt-parallel-corpus-eng-twi`](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi)
