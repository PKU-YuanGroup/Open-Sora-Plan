from huggingface_hub import hf_hub_download

# Define the repository and filename
repo_id = "LanguageBind/Open-Sora-Plan-v1.0.0"
filename = "vae/diffusion_pytorch_model.safetensors"

# Download the file
file_path = hf_hub_download(repo_id=repo_id, filename=filename)

print(f"File downloaded to {file_path}")