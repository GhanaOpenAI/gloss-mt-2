#!/usr/bin/env python3
"""Promote checkpoint-110000 of NLLB-600M to final and push to Hugging Face Hub."""

import os
import shutil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import HfApi

SRC_CKPT = "/mnt/volume_d2wey28/projects/gloss-mt-2/checkpoints/nllb-600m-en2tw/checkpoint-110000"
FINAL_DIR = "/mnt/volume_d2wey28/projects/gloss-mt-2/checkpoints/nllb-600m-en2tw/final"
REPO_ID = "ghanaopenai/nkrane-mt-nllb-600m-en2tw"
README_SRC = "/mnt/volume_d2wey28/projects/gloss-mt-2/repo/model_cards/nllb_README.md"

print(f"Loading model and tokenizer from {SRC_CKPT}...")
tok = AutoTokenizer.from_pretrained(SRC_CKPT)
model = AutoModelForSeq2SeqLM.from_pretrained(SRC_CKPT)

os.makedirs(FINAL_DIR, exist_ok=True)
print(f"Saving to {FINAL_DIR}...")
model.save_pretrained(FINAL_DIR)
tok.save_pretrained(FINAL_DIR)

# Copy README.md as model card
if os.path.isfile(README_SRC):
    shutil.copyfile(README_SRC, os.path.join(FINAL_DIR, "README.md"))
    print("Copied model card README.md into final directory.")

print(f"Uploading to Hugging Face Hub: {REPO_ID}...")
api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=FINAL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload fine-tuned Nkrane-MT NLLB-200 600M en->tw model (checkpoint-110000)",
)

print(f"Successfully uploaded model and tokenizer to https://huggingface.co/{REPO_ID}")
