# download_dataset.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

dataset = load_dataset("TIGER-Lab/WebInstruct-verified")
os.makedirs("dataset", exist_ok=True)
dataset.save_to_disk("./dataset/WebInstruct-verified")
print("Done. Saved to ./dataset/WebInstruct-verified")