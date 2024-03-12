
from huggingface_hub import snapshot_download
# snapshot_download(repo_id="liuhaotian/llava-v1.5-7b", local_dir="cache_dir/liuhaotian/llava-v1.5-7b", local_dir_use_symlinks=False, max_workers=1)
snapshot_download(repo_id="liuhaotian/llava-v1.6-vicuna-7b", local_dir="cache_dir/liuhaotian/llava-v1.6-vicuna-7b", local_dir_use_symlinks=False, max_workers=1)