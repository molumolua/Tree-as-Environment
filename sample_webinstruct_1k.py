# 从 WebInstruct-verified 中随机抽取 1k 条并保存为 parquet
from datasets import load_from_disk

ds = load_from_disk("./dataset/WebInstruct-verified")
# 若有多个 split，优先用 train，否则用第一个
if "train" in ds:
    full = ds["train"]
else:
    full = ds[list(ds.keys())[0]]

sampled = full.shuffle(seed=42).select(range(1000))
sampled.to_parquet("./dataset/webinstruct_1k.parquet")
print(f"Saved {len(sampled)} samples to ./dataset/webinstruct_1k.parquet")
