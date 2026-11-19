#!/usr/bin/env python3
"""Upload fine-tuned NLLB-200 1.3B final checkpoint and model card to Hugging Face Hub."""

import os
import shutil
from huggingface_hub import HfApi

FINAL_DIR = "/mnt/volume_d2wey28/projects/gloss-mt-2/repo/checkpoints/nllb-1.3b-en2tw/final"
REPO_ID = "ghanaopenai/nkrane-mt-nllb-1.3b-en2tw"
README_SRC = "/mnt/volume_d2wey28/projects/gloss-mt-2/repo/model_cards/nllb_1.3b_README.md"

if os.path.isfile(README_SRC):
    shutil.copyfile(README_SRC, os.path.join(FINAL_DIR, "README.md"))
    print("Copied model card README.md into final directory.")

print(f"Uploading {FINAL_DIR} to Hugging Face Hub: {REPO_ID}...")
api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=FINAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload fine-tuned Nkrane-MT NLLB-200 1.3B en->tw model (final converged)",
)

print(f"Successfully uploaded model and tokenizer to https://huggingface.co/{REPO_ID}")
