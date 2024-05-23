from huggingface_hub import hf_hub_download

file_path = hf_hub_download(repo_id="maxin-cn/Latte", filename="t2v.pt", revision="main")