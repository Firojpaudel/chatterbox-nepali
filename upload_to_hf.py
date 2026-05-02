import os
from huggingface_hub import HfApi

# Set HF_TOKEN env var before running: $env:HF_TOKEN="hf_..."
token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not set. Run: $env:HF_TOKEN='hf_...'")
repo_id = "Firoj112/chatterbox-nepali-runs"
api = HfApi()

files_to_upload = [
    ("README.md", "README.md"),
]

print(f"Starting upload to {repo_id}...")

for local_path, path_in_repo in files_to_upload:
    if os.path.exists(local_path):
        print(f"Uploading {local_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token
        )
        print(f"Uploaded {path_in_repo}")
    else:
        print(f"Skipping {local_path} (not found)")

print("All done!")
