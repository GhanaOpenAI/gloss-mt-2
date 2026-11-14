---
language:
- tw
- ak
- en
license: apache-2.0
base_model: Davlan/afro-xlmr-base
tags:
- cross-encoder
- reranker
- twi
- akan
- asante-twi
- afro-xlmr
- machine-translation
- lexical-disambiguation
- word-sense-disambiguation
- gloss-mt-2
- african-nlp
pipeline_tag: text-classification
metrics:
- accuracy
model-index:
- name: gloss-mt-2-reranker-afro-xlmr
  results:
  - task:
      type: text-classification
      name: Cross-Encoder Contextual Lexical Disambiguation
    dataset:
      name: gloss-mt-2-en-tw
      type: ghanaopenai/gloss-mt-2-en-tw
    metrics:
    - type: accuracy
      value: 97.21
---

# GLOSS-MT-2 Afro-XLMR Cross-Encoder Reranker

**`ghanaopenai/gloss-mt-2-reranker-afro-xlmr`** is a specialized cross-encoder reranker fine-tuned from [`Davlan/afro-xlmr-base`](https://huggingface.co/Davlan/afro-xlmr-base) for **contextual lexical disambiguation** in English $\rightarrow$ Asante Twi (Akan) translation.

It is a core component of the **GLOSS-MT-2** translation paradigm developed by [GhanaOpenAI](https://github.com/GhanaOpenAI/gloss-mt-2).

---

## 💡 Role in the GLOSS-MT-2 Pipeline

In GLOSS-MT-2, generative seq2seq models (e.g. Marian `opus-mt-en-tw` or `nllb-200-distilled-600M`) are trained to output fluent Asante Twi while keeping targeted English noun phrases preserved in angle brackets `<noun>` (with all Twi determiners such as `no`, `yi`, `bi` outside the brackets).

After the model generates the sentence, an offline dictionary lookup provides candidate Asante Twi translations for each bracketed term. Because many words have multiple distinct senses (e.g., *"trade"* can mean *"dwadie"* [commerce] or *"adwuma"* [profession]), this cross-encoder evaluates each candidate in the exact generated Asante Twi context in $<5\text{ ms}$ and selects the most contextually appropriate translation.

```
                         [English Source]
   "They went to see how the trade is doing in the market."
                                |
                                v
               [GLOSS-MT-2 Seq2Seq Translator]
                                |
                                v
     "Wɔkɔhwɛɛ sɛnea <trade> no teɛ wɔ dwadibea hɔ."
                                |
     Dictionary candidates for 'trade': ["dwadie", "adwuma", "aguadi"]
                                |
                                v
     [Afro-XLMR Cross-Encoder Reranker (This Model)]
       Pair 1: ("Wɔkɔhwɛɛ sɛnea <trade> no teɛ...", "dwadie") -> 0.984
       Pair 2: ("Wɔkɔhwɛɛ sɛnea <trade> no teɛ...", "adwuma") -> 0.112
                                |
                                v
                      [Final Translation]
        "Wɔkɔhwɛɛ sɛnea dwadie no teɛ wɔ dwadibea hɔ."
```

---

## 📊 Model Details

- **Base Architecture:** [`Davlan/afro-xlmr-base`](https://huggingface.co/Davlan/afro-xlmr-base) (multilingual Afrocentric transformer encoder).
- **Task:** Binary Cross-Encoder Classification / Reranking ($s \in [0, 1]$).
- **Tokenizer:** Extended with `<` and `>` special markers to preserve bracket spans.
- **Input Pair:**
  - `text_a`: Asante Twi sentence with preserved `<noun>` phrase (e.g. `Wɔkɔhwɛɛ sɛnea <trade> no teɛ wɔ dwadibea hɔ.`).
  - `text_b`: Candidate Asante Twi dictionary translation (e.g. `dwadie`).
- **Output:** Sigmoid score representing the semantic and idiomatic fit of `text_b` inside `text_a`.
- **Validation Accuracy:** **`97.21%`** on held-out validation pairs.
- **Inference Latency:** $< 5\text{ ms}$ per candidate pair on GPU.

---

## 🚀 Quickstart & Inference

### 1. Minimal Standalone Usage with `transformers`

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "ghanaopenai/gloss-mt-2-reranker-afro-xlmr"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
model.eval()

# Sentence context with bracketed English noun
context = "Wɔkɔhwɛɛ sɛnea <trade> no teɛ wɔ Kejetia dwadibea hɔ."
candidates = ["dwadie", "adwuma", "aguadi"]

# Score all candidates simultaneously
pairs_a = [context] * len(candidates)
pairs_b = candidates

inputs = tokenizer(
    pairs_a,
    pairs_b,
    padding=True,
    truncation=True,
    max_length=160,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    logits = model(**inputs).logits.squeeze(-1)
    scores = torch.sigmoid(logits).tolist()

for cand, score in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True):
    print(f"Candidate: {cand:<10} | Confidence: {score:.4f}")

# Output:
# Candidate: dwadie     | Confidence: 0.9852
# Candidate: aguadi     | Confidence: 0.8124
# Candidate: adwuma     | Confidence: 0.0815
```

---

### 2. Integration with `gloss-mt-2` Python Package

```python
from gloss_mt_2 import GlossMT2Translator
from gloss_mt_2.seq2seq import Seq2SeqModelWrapper
from gloss_mt_2.reranker import ContextualReranker

# Load fine-tuned NLLB-200 seq2seq generator
seq2seq = Seq2SeqModelWrapper("ghanaopenai/gloss-mt-2-nllb-600m-en2tw", direction="en-tw")

# Load Afro-XLMR Cross-Encoder Reranker
reranker = ContextualReranker("ghanaopenai/gloss-mt-2-reranker-afro-xlmr")

# End-to-end pipeline with post-translation contextual disambiguation
translator = GlossMT2Translator(model=seq2seq, reranker=reranker, direction="en-tw")

res = translator.translate("The Ministry of Finance announced a new policy for the hospital.")
print(res["translation"])
# -> "Sikasɛm Asoeɛ no de nhyehyɛeɛ foforɔ maa ayaresabea no."
```

---

## 🏋️ Training Setup

- **Dataset:** 606,404 curated training pairs and 13,024 validation pairs built from the clean [`ghanaopenai/gloss-mt-2-en-tw`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-2-en-tw) parallel corpus and Ghanaian lexicons.
- **Hardware:** Single NVIDIA H200 (140 GB VRAM), using `bfloat16` and `TF32`.
- **Optimizer:** AdamW (`lr=1e-5`, cosine learning rate schedule, warmup steps = 500).
- **Batch Size:** 16 per device with 4 gradient accumulation steps (effective batch size = 64).
- **Loss Function:** Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`).
- **Evaluation Strategy:** Evaluated every 500 steps on held-out validation pairs with early stopping patience of 10.
- **Peak Validation Accuracy:** `97.21%` (loss `0.0872`).

---

## 📖 Citation & Repository

If you use this model or the GLOSS-MT-2 pipeline in your research, please cite:

```bibtex
@misc{gloss_mt_2_2026,
  author = {Owusu, S. and GhanaOpenAI Contributors},
  title = {GLOSS-MT-2: Contextual Lexical Disambiguation & Cross-Encoder Reranking for English-Asante Twi Translation},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/ghanaopenai/gloss-mt-2-reranker-afro-xlmr}}
}
```

- **GitHub Repository:** [`GhanaOpenAI/gloss-mt-2`](https://github.com/GhanaOpenAI/gloss-mt-2)
- **Base Dataset:** [`ghanaopenai/gloss-mt-2-en-tw`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-2-en-tw)
