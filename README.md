# Nkrane-MT: Entity-Grounded Machine Translation & Post-Translation Glossary Substitution

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-GhanaOpenAI%2Fnkrane--mt-black.svg)](https://github.com/GhanaOpenAI/nkrane-mt)
[![Dataset](https://img.shields.io/badge/Dataset-nkrane--mt--parallel--corpus--eng--twi-yellow.svg)](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi)
[![Reranker](https://img.shields.io/badge/Reranker-afro--xlmr--reranker-green.svg)](https://huggingface.co/ghanaopenai/gloss-mt-2-reranker-afro-xlmr)

**Nkrane-MT** (*Nkrane*, in Akan, signifies a message dispatch, communication, and report) is a next-generation machine translation paradigm for African languages, starting with English $\leftrightarrow$ Asante Twi (Akan).

Rather than forcing artificial dictionary injection into English before translation, Nkrane-MT models translate fluent natural context with **dual-sided entity grounding**:
- Targeted concrete noun phrases, compound titles, organizations, and demographic groups are marked with angle brackets `<...>` on the **English input**.
- The seq2seq translation model directly preserves these exact bracketed terms on the **Asante Twi target**, keeping all Twi determiners outside the brackets.
- Abstract nouns (*information, development, peace, justice*) are translated freely and naturally into Twi.
- At inference time, an offline dictionary and our fine-tuned Afro-XLMR cross-encoder reranker evaluate candidate translations in context ($<5\text{ ms}$) and perform seamless post-translation lexical disambiguation.

---

## 🏛️ Architecture Overview

```
                          [Raw English Source]
          "The president of Ghana visited the Ministry of Finance."
                                    │
                                    ▼
                          [NounPhraseMarker]
       "The <president of Ghana> visited the <Ministry of Finance>."
                                    │
                                    ▼
                        [Nkrane-MT Seq2Seq Model]
       "<president of Ghana> no kɔsra <Ministry of Finance> no."
                                    │
     Candidates:
       • 'president of Ghana' -> ["Ghana manpanin", "Ghana hene"]
       • 'Ministry of Finance' -> ["Sikasɛm Asoeɛ"]
                                    │
                                    ▼
                  [Afro-XLMR Cross-Encoder Reranker]
                    Scores candidates in Twi context (<5 ms)
                                    │
                                    ▼
                         [Final Fluent Twi]
          "Ghana manpanin no kɔsra Sikasɛm Asoeɛ no."
```

---

## 🚀 Quickstart

### Installation
```bash
git clone https://github.com/GhanaOpenAI/nkrane-mt.git
cd nkrane-mt
pip install -e .
```

### Python Inference with Post-Translation Glossary Substitution

```python
from nkrane_mt import GlossMT2Translator
from nkrane_mt.seq2seq import Seq2SeqModelWrapper
from nkrane_mt.reranker import ContextualReranker

# 1. Load fine-tuned Nkrane-MT translation model
model = Seq2SeqModelWrapper("ghanaopenai/gloss-mt-2-nllb-600m-en2tw", direction="en-tw")

# 2. Load Afro-XLMR Cross-Encoder Reranker (optional for multi-candidate scoring)
reranker = ContextualReranker("ghanaopenai/gloss-mt-2-reranker-afro-xlmr")

# 3. Initialize translator (no dictionary loaded by default for zero memory overhead)
translator = GlossMT2Translator(model=model, reranker=reranker, direction="en-tw")

# 4. Translate with custom domain glossary override:
custom_glossary = [
    ("regional hospital", ["mantam ayaresabea", "akwahosan bea"]),
    ("Ministry of Finance", "Sikasɛm Asoeɛ"),
]

res = translator.translate(
    "The president of Ghana visited the regional hospital.",
    custom_glossary=custom_glossary,
)

print("Marked English Input:", res["marked_input"])
# -> "The <president of Ghana> visited the <regional hospital>."

print("Raw Model Output:    ", res["raw_translation"])
# -> "<president of Ghana> no kɔsra <regional hospital> no."

print("Final Translation:   ", res["translation"])
# -> "Ghana manpanin no kɔsra mantam ayaresabea no."
```

---

## 📊 Dataset & Benchmarks

- **Grand Parallel Corpus:** [`ghanaopenai/nkrane-mt-parallel-corpus-eng-twi`](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi) (~774,000 sentence pairs across GHOK, Pristine, News, and JW).
- **Afro-XLMR Reranker:** [`ghanaopenai/gloss-mt-2-reranker-afro-xlmr`](https://huggingface.co/ghanaopenai/gloss-mt-2-reranker-afro-xlmr) (**97.21% validation accuracy** on contextual lexical disambiguation).

---

## 🛠️ Data Preparation & Training on H200

```bash
# 1. Build Nkrane-MT Grand Dataset (with 10-pair batch streaming)
./run_build_grand_dataset.sh

# 2. Train Afro-XLMR Cross-Encoder Reranker
./run_train_reranker.sh

# 3. Fine-Tune Translation Models
./run_train_opus_mt_en2tw.sh      # Helsinki-NLP Marian OPUS-MT
./run_train_nllb_600m_en2tw.sh     # Meta NLLB-200 600M
```

---

## 📜 License
Apache-2.0
