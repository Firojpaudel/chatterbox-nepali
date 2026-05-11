"""Script to download the dataset locally."""
import os
from huggingface_hub import snapshot_download

repo_id = "Firoj112/chatterbox-multilingual-data"
local_dir = "data/chatterbox-multilingual-data"

print(f"Starting download of {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Download complete!")
