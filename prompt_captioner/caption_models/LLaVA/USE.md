
### Quick Start  

```
HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1   --master_port=29501  infer.py --train_file mini_data.json \
--caption_save_dir   llava_caption_dir --model_path  cache_dir/liuhaotian/llava-v1.5-7b \
--query 'describe this image in detail' --load-8bit --batch_size 10
```

`vid_batch_size`: used to preprocess video loading

`batch_size`: **true batch_size** during model inferring


