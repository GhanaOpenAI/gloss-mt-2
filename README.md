# GLOSS-MT-2: Source-Noun Preservation & Post-Translation Glossary Substitution

**GLOSS-MT-2** is a second-generation machine translation paradigm for low-resource languages, starting with English and Asante Twi (Akan).

Rather than requiring pre-translated dictionaries prior to training and pre-translating source inputs in-place, GLOSS-MT-2 models are trained to **translate fluent context while leaving key noun phrases exactly as they appear in the source text**.

At inference time, a post-translation dictionary substitution step detects the preserved source nouns and substitutes them with their exact domain translations—allowing dictionaries and custom glossaries to be updated, swapped, or customized per-domain without fine-tuning or modifying model weights.

---

## 💡 Why GLOSS-MT-2?

| Dimension | GLOSS-MT (v1) | GLOSS-MT-2 (This Work) |
|---|---|---|
| **Input Text** | Pre-translated mixed text (`<twi_term>` injected into English) | **Clean, untouched source text** |
| **Dictionary Requirement** | Requires pre-translated dictionary before translation starts | **Post-training substitution**: dictionary can be supplied or swapped after generation |
| **Custom Glossaries** | Hard to adapt to new domains without retraining | **Zero retraining**: user supplies any custom dictionary at runtime |
| **Out-of-Vocabulary Entities** | Fragile if terms are missing from the offline lexicon | **Naturally preserved**: unfamiliar entities remain intact in translation |
| **Directionality** | English → Twi only | **Bidirectional**: English ↔ Twi |

---

## 🧠 Supported Models

1. **GLOSS-MT NLLB-200 600M** (`facebook/nllb-200-distilled-600M`):
   - Fine-tuned bidirectionally: **`en → tw`** and **`tw → en`**.
2. **GLOSS-MT OPUS-MT en-tw** (`Helsinki-NLP/opus-mt-en-tw`):
   - Fine-tuned on **`en → tw`**.

---

## 🚀 Quickstart

### Installation
```bash
git clone https://github.com/GhanaOpenAI/gloss-mt-2.git
cd gloss-mt-2
pip install -e .
```

### Python Inference with Post-Translation Glossary Substitution

```python
from gloss_mt_2 import GlossMT2Translator
from gloss_mt_2.seq2seq import Seq2SeqModelWrapper

# Load fine-tuned GLOSS-MT-2 model
model = Seq2SeqModelWrapper("ghanaopenai/gloss-mt-2-nllb-600m-en2tw", direction="en-tw")
translator = GlossMT2Translator(model=model, direction="en-tw")

# 1. Standard translation with default offline dictionary
res = translator.translate("The Ministry of Finance announced a new policy for the hospital.")
print("Raw Model Output (Preserved):", res["raw_translation"])
# -> "<Ministry of Finance> no de <new policy> foforɔ maa <hospital> no."
print("Final Translation (Substituted):", res["translation"])
# -> "Sikasɛm Asoeɛ no de nhyehyɛeɛ foforɔ maa ayaresabea no."

# 2. Translation with custom domain glossary override
custom_terms = {
    "hospital": "akwahosan bea",
    "policy": "mmara nhyehyɛe",
}
res_custom = translator.translate(
    "The Ministry of Finance announced a new policy for the hospital.",
    custom_glossary=custom_terms,
)
print("Custom Translation:", res_custom["translation"])
# -> "Sikasɛm Asoeɛ no de mmara nhyehyɛe foforɔ maa akwahosan bea no."
```

---

## 🛠️ Data Preparation & Training on H200

### 1. Build Preserved Dataset
```bash
./run_build_dataset.sh
```

### 2. Run Training
```bash
# Train OPUS-MT (en -> tw)
./run_train_opus_mt_en2tw.sh

# Train NLLB-200 600M (en -> tw)
./run_train_nllb_600m_en2tw.sh

# Train NLLB-200 600M (tw -> en)
./run_train_nllb_600m_tw2en.sh

# Or run the full queue:
./run_all_training.sh
```

---

## 📜 License
Apache-2.0
