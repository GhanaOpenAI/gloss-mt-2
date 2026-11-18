# Nkrane-MT Parallel Corpus: Data Provenance & Replicability Specification

This document records the exact origin, generation methodology, and translation provenance of the **769,594 sentence pairs** in the [Nkrane-MT Parallel Corpus](https://huggingface.co/datasets/ghanaopenai/nkrane-mt-parallel-corpus-eng-twi) (`ghanaopenai/nkrane-mt-parallel-corpus-eng-twi`).

For transparency, research integrity, and future replicability, the composition of the four core pillars is detailed below.

---

## 📊 Summary Breakdown

| Pillar Tag | Training Pairs | Share | Text Source | Translation / Synthesis Method |
| :--- | :---: | :---: | :--- | :--- |
| **`news`** | **227,543** | 29.8% | Ghanaian News Articles (`ghananlpcommunity/twi-english-paragraph-dataset_news`) | Machine translated using Meta **NLLB-200 3.3B** (`facebook/nllb-200-3.3B`) |
| **`jw`** | **194,856** | 25.5% | Watchtower Publications (`AfricaCorpusLoader` / JW African Corpus) | **Human translation** by native Akan linguists and translators |
| **`pristine_*`** | **320,944** | 42.0% | Synthesized across 4 balanced registers (`ghanaopenai/pristine-twi-english`) | Generated using **Google Gemini** with Ghanaian news text as domain reference |
| • *pristine_monologue* | *133,599* | *17.5%* | Formal speeches, radio broadcasts, and monologues | Gemini-synthesized parallel text |
| • *pristine_narrative* | *126,022* | *16.5%* | Narrative accounts, descriptive prose, and reports | Gemini-synthesized parallel text |
| • *pristine_dialogue* | *49,087* | *6.4%* | Conversational exchanges and interview Q&A | Gemini-synthesized parallel text |
| • *pristine_storyful* | *12,236* | *1.6%* | Cultural stories, proverbs, idioms, and folklore | Gemini-synthesized parallel text |
| **`ghok`** | **20,251** | 2.7% | Academic & scientific research papers (`ghananlpcommunity/ghok-parallel-en-twi`) | Machine translated using **Google Translate** |
| **TOTAL (Train)** | **763,594** | 100% | *(+ 3,000 validation + 3,000 test splits)* | **769,594 Total Pairs** |

---

## 🏛️ Pillar Details

### 1. Ghanaian News (`news` — 227,543 pairs)
- **Domain:** Modern Ghanaian current affairs, public finance, local governance, infrastructure, and politics.
- **Origin:** Sourced from major Ghanaian news reporting outlets and portals.
- **Translation Engine:** Translated from English into Asante Twi using Meta's **NLLB-200 3.3B** model (`facebook/nllb-200-3.3B`).

### 2. JW African Corpus (`jw` — 194,856 pairs)
- **Domain:** High-register Akan prose, moral philosophy, narrative literature, and community discourse.
- **Origin:** Publications from the Watch Tower Bible and Tract Society.
- **Translation Engine:** **100% Human-translated** by professional native Akan/Twi translation teams over multiple decades. Provides the gold-standard syntactic foundation for natural Asante Twi idioms and particles.

### 3. Pristine Twi-English (`pristine_*` — 320,944 pairs)
- **Domain:** High-diversity literary, colloquial, and narrative registers.
- **Origin & Method:** Synthesized using **Google Gemini**, prompted using authentic Ghanaian news articles and cultural text as contextual reference anchors to ensure realistic domain vocabulary.
- **Stylistic Balance:**
  - **Monologue:** Speeches, educational lectures, radio addresses.
  - **Narrative:** Long-form descriptive accounts and story paragraphs.
  - **Dialogue:** Multi-turn conversational exchanges with colloquial particles (`wate`, `enti`, `paa`).
  - **Storyful:** Traditional idioms, allegories, and moral stories.

### 4. GHOK Corpus (`ghok` — 20,251 pairs)
- **Domain:** Academic research papers, scientific abstracts, governance, and technical reports.
- **Origin:** Compiled from scholarly publications and academic repositories.
- **Translation Engine:** Translated using **Google Translate**.

---

## 🤖 Dual-Sided Entity Grounding Pipeline (Gemini 3.8 Live API)

Once the raw parallel pairs from the 4 pillars were accumulated and deduplicated (using SHA-1 hashing), every pair was processed through **Google Gemini 3.8 Live API** (in micro-batches of 10 pairs) using a strict computational linguistic prompt:

1. **Concrete Noun Targeting:**
   - Detects concrete noun phrases, compound titles (`president of Ghana`, `Ministry of Finance`, `Bank of Ghana governor`), organizations (`Telecel`), demographic groups (`Persons with disabilities`), locations (`Kumasi`, `Accra`), and physical objects (`hospital`, `car`, `road`, `gold`).
2. **Strict Abstract Noun Exclusion:**
   - Abstract and conceptual nouns (`information`, `development`, `history`, `peace`, `justice`, `freedom`, `patience`, `policy`, `economy`) are **strictly excluded** from bracketing. This forces translation models to learn natural, fluent Akan equivalents rather than awkward literal nominal injections.
3. **Dual-Sided Bracket Placement:**
   - English input (`input`): Wraps targeted entities in `<...>` with English determiners outside (`The <president of Ghana>`).
   - Twi target (`target`): Matches the exact English phrase in `<...>`, keeping all Twi determiners outside (`<president of Ghana> no`).
   - Clean references (`english` and `twi`): Clean unbracketed text for standard reference evaluation.

---

## 🔁 Replicability Checklist

To replicate or extend this corpus:
1. Scribe or download the 4 raw sources.
2. Apply the pack-to-paragraph chunking heuristic (`scripts/build_grand_dataset.py`, max 512 characters).
3. Run `python scripts/build_grand_dataset.py` with valid Gemini API credentials to execute the 10-pair live alignment worker.
4. Verify PyArrow typed schema (`input`, `target`, `english`, `twi`, `phrases: List[string]`, `source`).
