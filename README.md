# Nkrane-MT: High-Performance English $\leftrightarrow$ Asante Twi Machine Translation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-GhanaOpenAI%2Fnkrane--mt-black.svg)](https://github.com/GhanaOpenAI/nkrane-mt)
[![Dataset](https://img.shields.io/badge/Dataset-769k_Parallel_Corpus-yellow.svg)](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi)
[![NLLB-600M](https://img.shields.io/badge/Model-NLLB--200_600M-blue.svg)](https://huggingface.co/ghanaopenai/nkrane-mt-nllb-600m-en2tw)
[![OPUS-MT](https://img.shields.io/badge/Model-OPUS--MT-green.svg)](https://huggingface.co/ghanaopenai/nkrane-mt-opus-en2tw)
[![uMT5-Base](https://img.shields.io/badge/Model-uMT5--Base-red.svg)](https://huggingface.co/ghanaopenai/nkrane-mt-umt5-base-en2tw)

**Nkrane-MT** (*Nkrane*, in Akan, signifies a message dispatch, communication, and report) is an open-source machine translation ecosystem for English $\leftrightarrow$ Asante Twi (Akan), developed by [GhanaOpenAI](https://github.com/GhanaOpenAI).

The pipeline supports two operating modes:
1. **Direct Translation (Pure Twi, Zero Brackets):** High-throughput batched translation designed to saturate any GPU (A10G, T4, V100, A100, H100, H200) for large-scale dataset translation jobs (60–225 sentences/sec).
2. **Entity-Grounded Glossary Pipeline:** Dual-sided entity preservation (`<entity>`) combined with an offline dictionary and a cross-encoder reranker for strict terminology constraints.

---

## 🏆 Benchmark Leaderboard

Models fine-tuned on the **769,594-pair [Nkrane-MT Parallel Corpus](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi)**:

| Model | Architecture | Parameters | SacreBLEU | chrF2 | Validation Loss | Hugging Face Hub |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Meta NLLB-200 600M** 🥇 | Seq2Seq (Enc-Dec) | 600M | **31.57** | **53.34** | **0.6976** | [`ghanaopenai/nkrane-mt-nllb-600m-en2tw`](https://huggingface.co/ghanaopenai/nkrane-mt-nllb-600m-en2tw) |
| **Helsinki OPUS-MT** 🥈 | Marian Seq2Seq | 77M | **29.72** | **50.34** | **1.5510** | [`ghanaopenai/nkrane-mt-opus-en2tw`](https://huggingface.co/ghanaopenai/nkrane-mt-opus-en2tw) |
| **Google uMT5-Base** 🥉 | UniMax T5 | 580M | **22.01** | **41.75** | **1.3270** | [`ghanaopenai/nkrane-mt-umt5-base-en2tw`](https://huggingface.co/ghanaopenai/nkrane-mt-umt5-base-en2tw) |

Supporting models:
- **Inference Noun Phrase Marker:** [`ghanaopenai/nkrane-mt-marker-en`](https://huggingface.co/ghanaopenai/nkrane-mt-marker-en) (94.70% accuracy / 0.8139 F1)
- **Afro-XLMR Cross-Encoder Reranker:** [`ghanaopenai/nkrane-mt-reranker-afro-xlmr`](https://huggingface.co/ghanaopenai/nkrane-mt-reranker-afro-xlmr) (97.21% accuracy)

---

## 📦 Installation

```bash
git clone https://github.com/GhanaOpenAI/nkrane-mt.git
cd nkrane-mt
pip install -e .
```

---

## 🚀 High-Throughput Batch Translation (Large Dataset Jobs)

When translating large datasets (thousands to millions of sentences), use `translate_batch()` or `translate_stream()`. These bypass entity marking and dictionary overhead, running pure model generation in `torch.inference_mode()` with dynamic batch padding to **fully saturate GPU compute and VRAM**.

### Python API

```python
from nkrane_mt.inference import NkraneMTTranslator

# Load any of our fine-tuned models (e.g. NLLB-600M or OPUS-MT)
translator = NkraneMTTranslator(
    model="ghanaopenai/nkrane-mt-nllb-600m-en2tw",
    device="cuda:0",
)

sentences = [
    "The president of Ghana visited the military hospital in Accra yesterday.",
    "Gold and cocoa are major exports of Ghana.",
    "The doctor gave medicine to the sick child.",
    "Farmers in the Ashanti region need better roads to transport their produce."
]

# High-speed batch translation (no brackets, complete Twi)
twi_translations = translator.translate_batch(
    sentences,
    batch_size=64, # Increase to 128/256 for A100/H100/H200
)

for en, tw in zip(sentences, twi_translations):
    print(f"EN : {en}")
    print(f"TWI: {tw}\n")
```

### CLI for Translating Files (Parquet, JSONL, CSV, or Hugging Face Datasets)

Use the built-in streaming dataset translator CLI to process large datasets without running out of memory:

```bash
# 1. Translate a Hugging Face Dataset and stream to Parquet:
python scripts/translate_dataset.py \
    --model ghanaopenai/nkrane-mt-nllb-600m-en2tw \
    --input databricks/databricks-dolly-15k \
    --split train \
    --src-column instruction \
    --tgt-column instruction_twi \
    --output dolly_twi.parquet \
    --batch-size 64

# 2. High-speed translation of local Parquet/JSONL files:
python scripts/translate_dataset.py \
    --model ghanaopenai/nkrane-mt-opus-en2tw \
    --input my_corpus.parquet \
    --src-column text \
    --tgt-column twi_text \
    --output my_corpus_translated.parquet \
    --batch-size 128
```

---

## 🔬 Entity-Grounded Pipeline with Glossary Substitution

For enterprise or specialized translation where specific terminology must strictly match a custom dictionary:

```python
from nkrane_mt.inference import NkraneMTTranslator

translator = NkraneMTTranslator(
    model="ghanaopenai/nkrane-mt-nllb-600m-en2tw",
    marker_model_name="ghanaopenai/nkrane-mt-marker-en",
    use_default_dictionary=True, # Resolves entities via offline Twi dictionary
)

# Custom domain terminology override:
custom_glossary = {
    "regional hospital": "mantam ayaresabea",
    "Ministry of Finance": "Sikasɛm Asoeɛ",
}

res = translator.translate(
    "The president of Ghana visited the regional hospital yesterday.",
    custom_glossary=custom_glossary,
)

print(res["marked_input"])
# -> "The <president of Ghana> visited the <regional hospital> yesterday."

print(res["raw_translation"])
# -> "<president of Ghana> no kɔsraa <regional hospital> no ɛnnɛra."

print(res["translation"])
# -> "Ghana manpanyin no kɔsraa mantam ayaresabea no ɛnnɛra."
```

---

## 🏛️ Dataset Provenance & 4 Core Pillars

The **769,594-pair [Nkrane-MT Parallel Corpus](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi)** was built across four balanced linguistic pillars (full details in [`docs/data_provenance.md`](docs/data_provenance.md)):

| Pillar | Training Pairs | Share | Domain & Origin | Translation / Generation Method |
| :--- | :---: | :---: | :--- | :--- |
| **`news`** | **227,543** | 29.8% | Ghanaian News (`twi-english-paragraph-dataset_news`) | Machine translated using Meta **NLLB-200 3.3B** |
| **`jw`** | **194,856** | 25.5% | Watchtower Akan Corpus (`AfricaCorpusLoader`) | **100% Human translation** by native Akan linguists |
| **`pristine_*`** | **320,944** | 42.0% | Literary & dialogue registers (`pristine-twi-english`) | Synthesized with **Google Gemini** using news as domain anchor |
| **`ghok`** | **20,251** | 2.7% | Academic & scientific papers (`ghok-parallel-en-twi`) | Machine translated using **Google Translate** |

*All 769k pairs were processed through **Gemini 3.8 Live API** (batches of 10) for dual-sided entity grounding with strict abstract noun filtering.*

---

## 🛠️ Training Scripts (100% Portable)

All training runners automatically download the dataset from Hugging Face if no local cache is found:

```bash
./run_train_nllb_600m_en2tw.sh    # Meta NLLB-200 600M
./run_train_opus_mt_en2tw.sh      # Helsinki Marian OPUS-MT
./run_train_umt5_base_en2tw.sh    # Google uMT5-Base
./run_train_marker.sh             # Noun Phrase Marker
./run_train_reranker.sh           # Afro-XLMR Contextual Reranker
```

---

## 📜 License
Apache-2.0
