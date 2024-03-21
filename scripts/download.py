from huggingface_hub import HfApi, snapshot_download
import os 

# download models and cache (from hub)
local_dir = os.getcwd().split("scripts/")[0]

snapshot_download(repo_id="yashkant/spad", local_dir=local_dir)

