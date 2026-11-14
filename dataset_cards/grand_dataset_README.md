---
language:
- en
- tw
- ak
license: cc-by-nc-4.0
task_categories:
- translation
tags:
- twi
- akan
- asante-twi
- english
- machine-translation
- parallel-corpus
- gloss-mt-2
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

# GLOSS-MT-2: Grand Multi-Pillar English → Asante Twi (Akan) Parallel Dataset

**`ghanaopenai/gloss-mt-2-en-tw-grand`** is a massive, balanced, sentence-aligned parallel corpus for training second-generation GLOSS-MT models (**GLOSS-MT-2**).

Rather than forcing artificial dictionary term injection before translation, GLOSS-MT-2 trains models on clean English inputs to generate fluent Asante Twi while **preserving key noun phrases in angle brackets `<...>`**, keeping all Twi determiners outside the brackets. Post-translation substitution and contextual lexical disambiguation are handled by an offline dictionary and an Afro-XLMR cross-encoder reranker.

---

## 🏛️ The Four Pillars of the Grand Corpus

The dataset draws from four distinct corpora to ensure comprehensive coverage across formal, colloquial, narrative, and technical registers:

| Pillar | Source | Coverage & Style |
|---|---|---|
| **1. GHOK Parallel Corpus** | `ghananlpcommunity/ghok-parallel-en-twi` | STEM, technical, academic, health, and governance text |
| **2. Pristine Twi-English** | `ghanaopenai/pristine-twi-english` | Balanced evenly across **Narrative (25%)**, **Dialogue (25%)**, **Storyful (25%)**, and **Monologue (25%)** |
| **3. Ghanaian News Paragraphs** | `ghananlpcommunity/twi-english-paragraph-dataset_news` | Modern Ghanaian current affairs, local politics, economy, and community news |
| **4. JW African Corpus** | Verse-aligned Akan Literature | Fluent everyday Akan literature, idiomatic expressions, and conversational dialogue |

---

## 🎯 Selective Noun Targeting Rules

To eliminate the unnatural syntax of legacy dictionary-injection datasets, targeting strictly follows three rules:

1. **Multi-Word Noun Phrases ($\ge 2$ words):**
   Compound nouns and entities are targeted, including alphanumeric phrases:
   - `<Ministry of Finance>`, `<regional hospital>`, `<Kejetia market place>`, `<May 2026>`, `<high court>`, `<small afternoon>`.
2. **Capitalized Single-Word Entities (Named Entities / Proper Nouns):**
   - `<Ghana>`, `<Telecel>`, `<Opoku>`, `<Jesus>`, `<Kumasi>`, `<Accra>`.
3. **Common Lowercase Single-Word Nouns are NOT Bracketed:**
   - Everyday single-word nouns (`man`, `soldier`, `matter`, `trade`, `nation`, `water`, `house`) are left unbracketed so seq2seq models translate them naturally into Twi without forced glossary substitution.
4. **Grammar & Determiners Stay Outside Brackets:**
   - All Twi articles and determiners (`no`, `yi`, `bi`, `mu`, `ho`) stay in Twi outside the brackets:
     - `saa <matter> yi` *(not `<this matter>`)*
     - `<government> no` *(not `<the government>`)*
     - `wɔ <May 2026> mu`
5. **Length & Punctuation Guard:**
   - 100% complete grammatical sentences $\le 512$ characters (never broken across sentences). All markdown formatting (`|`, `---`, `#`, `>`) is stripped.

---

## 📋 Schema & Examples

Each record contains 5 fields:

| Field | Type | Description |
|---|---|---|
| **`input`** | `string` | Raw natural English sentence (no brackets, no preprocessing). |
| **`target`** | `string` | Fluent Asante Twi with targeted noun phrases preserved in `<...>` (determiners outside). |
| **`english`** | `string` | Clean English reference sentence. |
| **`twi`** | `string` | Clean human reference Asante Twi translation. |
| **`source`** | `string` | Origin pillar (`ghok`, `pristine_narrative`, `pristine_dialogue`, `news`, `jw`). |

### Example 1: Multi-word technical entity & dates
- **Input (`input`):**
  > `The president of Ghana visited the high court in May 2026 to open the new session.`
- **Target (`target`):**
  > `Ghana <president> no kɔsra <high court> no wɔ <May 2026> mu sɛ ɔrebɛbue nhyiamu foforɔ no.`
- **Reference Twi (`twi`):**
  > `Ghana manpanin no kɔsra asɛnnibea kɛseɛ no wɔ May 2026 mu sɛ ɔrebɛbue nhyiamu foforɔ no.`

### Example 2: Everyday narrative with corporate and community entities
- **Input (`input`):**
  > `Telecel Ghana announced that they will build a regional hospital near the market place.`
- **Target (`target`):**
  > `<Telecel Ghana> de bɔmmɔdie too gua sɛ wɔbɛsi <regional hospital> foforɔ bi a ɛbɛn dwadibea hɔ.`
- **Reference Twi (`twi`):**
  > `Telecel Ghana de bɔmmɔdie too gua sɛ wɔbɛsi mantam ayaresabea foforɔ bi a ɛbɛn dwadibea hɔ.`

---

## 🔬 Dataset Splits & Alignment Status

- **Alignment Engine:** Google Gemini 3.8 Live API via 36 concurrent WebSocket workers across 12 pooled API keys.
- **Total Chunks in Corpus:** 763,094 complete sentences.
- **Live Snapshots:** Incremental snapshots are pushed to this Hugging Face repository during alignment.
- **Split Distribution:**
  - `train`: 90%+
  - `validation`: 3,000 held-out sentence pairs
  - `test`: 3,000 benchmark sentence pairs

---

## 🛠️ Usage with `datasets`

```python
from datasets import load_dataset

# Load GLOSS-MT-2 Grand Dataset
ds = load_dataset("ghanaopenai/gloss-mt-2-en-tw-grand")

print("Sample training pair:")
print("Input: ", ds["train"][0]["input"])
print("Target:", ds["train"][0]["target"])
```

---

## 📖 Related Models & Repositories

- **GitHub Repository:** [`GhanaOpenAI/gloss-mt-2`](https://github.com/GhanaOpenAI/gloss-mt-2)
- **Afro-XLMR Cross-Encoder Reranker:** [`ghanaopenai/gloss-mt-2-reranker-afro-xlmr`](https://huggingface.co/ghanaopenai/gloss-mt-2-reranker-afro-xlmr)
- **Baseline Dataset:** [`ghanaopenai/gloss-mt-2-en-tw`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-2-en-tw)
