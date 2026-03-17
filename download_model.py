# 使用 hf-mirror 下载 Hugging Face 模型到本地 ./model
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

def main():
    model_id = "TIGER-Lab/general-verifier"
    local_dir = "./model"
    os.makedirs(local_dir, exist_ok=True)
    path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {path}")

if __name__ == "__main__":
    main()
