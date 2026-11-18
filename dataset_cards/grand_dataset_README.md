---
language:
- en
- tw
- ak
license: cc-by-nc-4.0
task_categories:
- translation
tags:
- nkrane-mt
- nkrane
- twi
- akan
- asante-twi
- english
- machine-translation
- parallel-corpus
- entity-grounded
- african-nlp
- ghana
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Nkrane-MT Parallel Corpus: Entity-Grounded English ↔ Asante Twi (Akan) Parallel Dataset

**`ghanaopenai/nkrane-mt-parallel-corpus-eng-twi`** is a landmark, balanced, 774,000-sentence parallel corpus for training the **Nkrane-MT** (*Nkrane*, in Akan, signifies a message dispatch, communication, and report) machine translation pipeline developed by [GhanaOpenAI](https://github.com/GhanaOpenAI/nkrane-mt).

Rather than forcing artificial dictionary term injection before translation, **Nkrane-MT** models are trained with dual-sided bracketed entity preservation:
1. **English Input (`input`):** Contains targeted concrete noun phrases, compound titles, organizations, places, and demographic groups in angle brackets `<...>` with English determiners outside:
   > `The <president of Ghana> visited the <Ministry of Finance> in <May 2026>.`
2. **Twi Target (`target`):** Contains the exact same English noun phrases inside `<...>`, with all Asante Twi determiners and grammar particles outside:
   > `<president of Ghana> no kɔsra <Ministry of Finance> no wɔ <May 2026> mu.`
3. **Reference English (`english`):** Clean, unbracketed source text:
   > `The president of Ghana visited the Ministry of Finance in May 2026.`
4. **Reference Twi (`twi`):** Clean, unbracketed reference Asante Twi translation:
   > `Ghana manpanin no kɔsra Sikasɛm Asoeɛ no wɔ May 2026 mu.`

Post-translation lexical substitution and contextual word-sense disambiguation are handled at inference time by an offline dictionary and our fine-tuned Afro-XLMR cross-encoder reranker ([`ghanaopenai/nkrane-mt-reranker-afro-xlmr`](https://huggingface.co/ghanaopenai/nkrane-mt-reranker-afro-xlmr)).

---

## 🏛️ The Four Pillars of the Nkrane-MT Corpus (Data Provenance)

The dataset draws from four distinct corpora to ensure comprehensive coverage across formal, colloquial, narrative, and technical registers. For full research transparency and replicability:

| Pillar | Training Pairs | Share | Domain & Source | Translation / Generation Method |
|---|:---:|:---:|---|---|
| **1. Ghanaian News (`news`)** | **227,543** | 29.8% | Ghanaian news articles, politics, infrastructure, and current affairs (`ghananlpcommunity/twi-english-paragraph-dataset_news`) | Machine translated using Meta **NLLB-200 3.3B** (`facebook/nllb-200-3.3B`) |
| **2. JW African Corpus (`jw`)** | **194,856** | 25.5% | High-register literature, historical accounts, and moral philosophy (`AfricaCorpusLoader` / JW) | **100% Human translation** by professional native Akan translators |
| **3. Pristine Twi-English (`pristine_*`)** | **320,944** | 42.0% | Balanced evenly across **Monologue (17.5%)**, **Narrative (16.5%)**, **Dialogue (6.4%)**, and **Storyful (1.6%)** (`ghanaopenai/pristine-twi-english`) | Synthesized using **Google Gemini** using authentic Ghanaian news articles as domain reference |
| **4. GHOK Parallel Corpus (`ghok`)** | **20,251** | 2.7% | Academic research papers, scientific abstracts, and governance (`ghananlpcommunity/ghok-parallel-en-twi`) | Machine translated using **Google Translate** |
| **TOTAL** | **763,594** | 100% | *(+ 3,000 validation + 3,000 test splits)* | **769,594 Total Pairs** |

---

## 🎯 Selective Noun Targeting Rules

To eliminate the unnatural syntax of legacy dictionary-injection datasets and preserve natural Akan semantics:

1. **Abstract Nouns are EXCLUDED (Translated Freely into Twi):**
   - Abstract nouns and conceptual terms (`information`, `development`, `history`, `investigation`, `peace`, `justice`, `freedom`, `decision`, `beauty`, `patience`, `knowledge`, `policy`) are **NEVER bracketed or forced into English**. Models translate them naturally, idiomatically, and freely into Asante Twi.
2. **Concrete (Non-Abstract) Noun Phrases are INCLUDED (Even Single Words):**
   - Tangible, physical objects, entities, and institutions (`hospital`, `regional hospital`, `car`, `road`, `soldier`, `doctor`, `gold`, `water`, `phone`, `computer`, `court`, `market place`, `airport`) are included, whether single-word or multi-word.
3. **Complex Nominal Complements & Demographic Groups:**
   - Prepositional nominal complements are captured as complete units (`president of Ghana`, `Ministry of Finance`, `Bank of Ghana governor`, `Minister for Health`, `King of Asante`, `Persons with disabilities`).
4. **Capitalized Nouns & Proper Entities:**
   - Capitalized entities appearing inside the sentence are preserved (`Ghana`, `Telecel`, `Opoku`, `Jesus`, `Ministry of Finance`, `May 2026`).
   - **Sentence-Initial Capitalization Guard:** If a noun phrase begins the sentence, it is preserved if it is **multi-word** (`Ministry of Finance...`, `Telecel Ghana...`). If it is a **single word** (`Information...`, `Development...`), capitalization is treated as grammatical sentence-initial casing, so the abstractness filter applies: it is kept only if concrete/non-abstract (`Hospital...`, `Ghana...`) and excluded if abstract (`Information...`, `Development...`).
5. **Grammar & Determiners Stay Outside Brackets:**
   - All Twi articles and determiners (`no`, `yi`, `bi`, `mu`, `ho`) stay in Twi outside the brackets:
     - `saa <hospital> yi` *(not `<this hospital>`)*
     - `<court> no` *(not `<the court>`)*
     - `wɔ <May 2026> mu`
6. **Length & Sentence Boundary Guard:**
   - 100% complete grammatical sentences $\le 512$ characters (never broken across sentences). All markdown formatting (`|`, `---`, `#`, `>`) is stripped.

---

## 📋 Schema & Examples

Each record contains 6 fields:

| Field | Type | Description |
|---|---|---|
| **`input`** | `string` | English sentence with targeted noun phrases in `<...>` (determiners outside). |
| **`target`** | `string` | Fluent Asante Twi with the exact same English noun phrases in `<...>` (determiners outside). |
| **`english`** | `string` | Clean raw English reference sentence (for inference marker model training). |
| **`twi`** | `string` | Clean human reference Asante Twi translation. |
| **`phrases`** | `list[str]` | List of the bracketed phrases. |
| **`source`** | `string` | Origin pillar (`ghok`, `pristine_narrative`, `pristine_dialogue`, `news`, `jw`). |

### Example 1: Multi-word technical entity & dates
- **Input (`input`):**
  > `The <president of Ghana> visited the <high court> in <May 2026> to open the new session.`
- **Target (`target`):**
  > `<president of Ghana> no kɔsra <high court> no wɔ <May 2026> mu sɛ ɔrebɛbue nhyiamu foforɔ no.`
- **Reference English (`english`):**
  > `The president of Ghana visited the high court in May 2026 to open the new session.`
- **Reference Twi (`twi`):**
  > `Ghana manpanin no kɔsra asɛnnibea kɛseɛ no wɔ May 2026 mu sɛ ɔrebɛbue nhyiamu foforɔ no.`

### Example 2: Demographic group & abstract concept translation
- **Input (`input`):**
  > `<Persons with disabilities> are an important part of any economy.`
- **Target (`target`):**
  > `<Persons with disabilities> no yɛ fa a ɛho hia wɔ sikasɛm biara mu.`
- **Reference English (`english`):**
  > `Persons with disabilities are an important part of any economy.`
- **Reference Twi (`twi`):**
  > `Nnipa a wɔwɔ dɛm yɛ fa a ɛho hia wɔ sikasɛm biara mu.`

---

## 🔬 Dataset Splits & Alignment Engine

- **Alignment Engine:** Google Gemini 3.8 Live API via 36 concurrent WebSocket workers across 12 pooled API keys with 10-pair batch streaming.
- **Total Corpus Size:** **769,594 sentence pairs** (100% complete grammatical sentences $\le 512$ chars).
- **Split Distribution:**
  - `train`: **763,594 sentence pairs**
  - `validation`: **3,000 held-out sentence pairs**
  - `test`: **3,000 benchmark sentence pairs**

---

## 🛠️ Usage with `datasets`

```python
from datasets import load_dataset

# Load Nkrane-MT Grand Parallel Corpus
ds = load_dataset("ghanaopenai/nkrane-mt-parallel-corpus-eng-twi")

print("Sample training pair:")
print("Input (Bracketed English):", ds["train"][0]["input"])
print("Target (Bracketed Twi):   ", ds["train"][0]["target"])
print("Clean Reference English:  ", ds["train"][0]["english"])
print("Clean Reference Twi:      ", ds["train"][0]["twi"])
```

---

## 📖 Citation & Repository

```bibtex
@misc{nkrane_mt_2026,
  author = {Owusu, S. and GhanaOpenAI Contributors},
  title = {Nkrane-MT: Entity-Grounded English-Asante Twi (Akan) Parallel Corpus},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi}}
}
```

- **GitHub Repository:** [`GhanaOpenAI/nkrane-mt`](https://github.com/GhanaOpenAI/nkrane-mt)
- **Afro-XLMR Cross-Encoder Reranker:** [`ghanaopenai/nkrane-mt-reranker-afro-xlmr`](https://huggingface.co/ghanaopenai/nkrane-mt-reranker-afro-xlmr)
