from huggingface_hub import snapshot_download

# Replace "your_token_here" with the token you copied
hf_token = "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"
repo_id = "sentence-transformers-testing/stsb-bert-tiny-onnx"
local_dir = "./stsb-bert-tiny-onnx"  # Specify where to download the files

# Authenticate and download the repository
snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token)


