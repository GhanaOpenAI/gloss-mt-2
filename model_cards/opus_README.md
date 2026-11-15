---
language:
- en
- tw
license: apache-2.0
base_model: Helsinki-NLP/opus-mt-en-tw
tags:
- translation
- seq2seq
- machine-translation
- akan
- twi
- asante-twi
- nkrane-mt
- african-nlp
pipeline_tag: translation
metrics:
- bleu
- chrf
model-index:
- name: nkrane-mt-opus-en2tw
  results:
  - task:
      type: translation
      name: English to Asante Twi Translation
    dataset:
      name: nkrane-mt-parallel-corpus-eng-twi
      type: ghanaopenai/nkrane-mt-parallel-corpus-eng-twi
    metrics:
    - type: bleu
      value: 29.72
      name: SacreBLEU (Substituted)
    - type: chrf
      value: 50.34
      name: chrF2 (Substituted)
    - type: bleu
      value: 24.63
      name: SacreBLEU (Raw Preserved)
---

# Nkrane-MT OPUS-MT (`nkrane-mt-opus-en2tw`)

**`ghanaopenai/nkrane-mt-opus-en2tw`** is an entity-grounded English $\to$ Asante Twi translation model based on `Helsinki-NLP/opus-mt-en-tw`, fine-tuned on the 769,594-pair [Nkrane-MT Parallel Corpus](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi) using dual-sided entity bracket preservation.

Part of the [**Nkrane-MT**](https://github.com/GhanaOpenAI/nkrane-mt) pipeline developed by [GhanaOpenAI](https://github.com/GhanaOpenAI).

---

## 🎯 Benchmark Performance

Evaluated on held-out test splits with live SacreBLEU, chrF2, and post-translation dictionary substitution:

| Metric | Substituted (Final Output) | Raw (Bracketed Generation) |
| :--- | :---: | :---: |
| **SacreBLEU** | **29.72** | 24.63 |
| **chrF2** | **50.34** | 46.95 |
| **Evaluation Loss** | **1.551** | - |

*Note: The **+5.09 BLEU gain** from Raw to Substituted reflects the pipeline's offline dictionary and reranker resolving preserved English entity brackets into natural Twi.*

---

## 🚀 Usage with `nkrane_mt`

```bash
pip install git+https://github.com/GhanaOpenAI/nkrane-mt.git
```

```python
from nkrane_mt.inference import GlossMT2Translator

translator = GlossMT2Translator(
    model_name="ghanaopenai/nkrane-mt-opus-en2tw",
    marker_model_name="ghanaopenai/nkrane-mt-marker-en",
    direction="en-tw",
)

result = translator.translate("The president of Ghana visited the military hospital in Accra yesterday.")
print(result["final_translation"])
# -> Ɔmampanyin a ɔwɔ Ghana kɔsra asraafoɔ ayaresabea a ɛwɔ Nkran nnera.
```

---

## 🏗️ Architecture & Training

- **Base Architecture:** Marian Seq2Seq LM (`Helsinki-NLP/opus-mt-en-tw`)
- **Dataset:** 763,594 training pairs ([`ghanaopenai/nkrane-mt-parallel-corpus-eng-twi`](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi))
- **Precision:** `bfloat16` + `TF32` on NVIDIA H200
- **Batch Size:** 32 per device $\times$ 2 gradient accumulation steps (effective batch size 64)
- **Learning Rate:** 1e-4 with AdamW and linear warmup
- **Selected Checkpoint:** Step 122,000 (Epoch 10.22, optimal composite score)

---

## 📄 Citation

```bibtex
@misc{nkrane-mt-2026,
  author = {GhanaOpenAI},
  title = {Nkrane-MT: Entity-Grounded English-Twi Translation Pipeline},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/ghanaopenai/nkrane-mt-opus-en2tw}}
}
```
