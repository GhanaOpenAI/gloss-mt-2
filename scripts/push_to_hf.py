"""Push GLOSS-MT-2 datasets to Hugging Face under GhanaOpenAI."""

import os
from datasets import load_from_disk
from huggingface_hub import HfApi

data_dir = os.environ.get("DATA_DIR", "build/preserved-dataset")
api = HfApi()

CARD_EN2TW = """---
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
- machine-translation
- gloss-mt
- gloss-mt-2
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

# GLOSS-MT-2: English → Asante Twi (Akan) Parallel Dataset

Parallel dataset for training **GLOSS-MT-2** models.

### Paradigm
- **Input (`source`):** Raw, un-preprocessed English text (no tags, brackets, or pre-translation).
- **Target (`target`):** Fluent Asante Twi translation where English noun phrases are left untranslated and enclosed in angle brackets `<...>` for post-translation glossary substitution.
- **Reference (`gold`):** Clean human Asante Twi translation (for metric computation).
- **Preserved Terms (`terms`):** Aligned noun phrase pairs.

### Statistics
| Split | Total Chunks | With Preserved Nouns | Preserved Terms |
|---|---|---|---|
| **train** | 31,909 | 30,010 (94.0%) | 558,253 |
| **validation** | 701 | 667 (95.1%) | 12,330 |
| **test** | 1,100 | 1,043 (94.8%) | 20,230 |

### Sample
- **Source (English):** `## Mawuli Bruce Spetua, a man who is twenty-seven years old, and he has a UCC certificate, and he wants to serve this nation as a soldier.`
- **Target (Twi + `<En>`):** `## <Mawuli Bruce Spetua>, <man> a wadi mfeɛ aduonu nson, na ɔwɛ <UCC certificate>, na ɔpɛ sɛ ɔkɔsom <nation> yi sɛ <soldier>.`
- **Gold (Clean Twi):** `## Mawuli Bruce Spetua, ɔbarima a wadi mfeɛ aduonu nson, na ɔwɛ UCC abodin krataa, na ɔpɛ sɛ ɔkɔsom ɔman yi sɛ asrafoɔ.`

### Code Repository
[`GhanaOpenAI/gloss-mt-2`](https://github.com/GhanaOpenAI/gloss-mt-2)
"""

CARD_TW2EN = """---
language:
- tw
- ak
- en
license: cc-by-nc-4.0
task_categories:
- translation
tags:
- twi
- akan
- asante-twi
- machine-translation
- gloss-mt
- gloss-mt-2
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

# GLOSS-MT-2: Asante Twi (Akan) → English Parallel Dataset

Parallel dataset for training **GLOSS-MT-2** models.

### Paradigm
- **Input (`source`):** Raw, un-preprocessed Asante Twi text (no tags, brackets, or pre-translation).
- **Target (`target`):** Fluent English translation where Twi noun phrases are left untranslated and enclosed in angle brackets `<...>` for post-translation glossary substitution.
- **Reference (`gold`):** Clean human English translation (for metric computation).
- **Preserved Terms (`terms`):** Aligned noun phrase pairs.

### Statistics
| Split | Total Chunks | With Preserved Nouns | Preserved Terms |
|---|---|---|---|
| **train** | 31,909 | 30,010 (94.0%) | 558,253 |
| **validation** | 701 | 667 (95.1%) | 12,330 |
| **test** | 1,100 | 1,043 (94.8%) | 20,230 |

### Sample
- **Source (Twi):** `## Mawuli Bruce Spetua, ɔbarima a wadi mfeɛ aduonu nson, na ɔwɛ UCC abodin krataa, na ɔpɛ sɛ ɔkɔsom ɔman yi sɛ asrafoɔ.`
- **Target (English + `<Tw>`):** `## <Mawuli Bruce Spetua>, a <ɔbarima> who is twenty-seven years old, and he has a <UCC abodin krataa>, and he wants to serve this <ɔman> as a <asrafoɔ>.`
- **Gold (Clean English):** `## Mawuli Bruce Spetua, a man who is twenty-seven years old, and he has a UCC certificate, and he wants to serve this nation as a soldier.`

### Code Repository
[`GhanaOpenAI/gloss-mt-2`](https://github.com/GhanaOpenAI/gloss-mt-2)
"""


def main():
    # 1. Push EN -> TW
    repo_en2tw = "ghanaopenai/gloss-mt-2-en-tw"
    print(f"Pushing {repo_en2tw}...")
    ds_en2tw = load_from_disk(os.path.join(data_dir, "en_to_tw"))
    ds_en2tw.push_to_hub(repo_en2tw)
    api.upload_file(
        path_or_fileobj=CARD_EN2TW.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_en2tw,
        repo_type="dataset",
    )
    print(f"Pushed {repo_en2tw} and uploaded dataset card!")

    # 2. Push TW -> EN
    repo_tw2en = "ghanaopenai/gloss-mt-2-tw-en"
    print(f"\nPushing {repo_tw2en}...")
    ds_tw2en = load_from_disk(os.path.join(data_dir, "tw_to_en"))
    ds_tw2en.push_to_hub(repo_tw2en)
    api.upload_file(
        path_or_fileobj=CARD_TW2EN.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_tw2en,
        repo_type="dataset",
    )
    print(f"Pushed {repo_tw2en} and uploaded dataset card!")


if __name__ == "__main__":
    main()
