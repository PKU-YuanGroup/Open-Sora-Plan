
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch \
    --config_file scripts/accelerate_configs/ddp_config.yaml --main_process_port 29502 \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --ae stabilityai/sd-vae-ft-mse \
    --data-path /root/autodl-tmp/sea.csv \
    --video-folder /root/autodl-tmp/sea \
    --text-encoder-name DeepFloyd/t5-v1_1-xxl \
    --model-max-length 120 \
    --dataset t2v \
    --extras 3 \
    --sample-rate 1 \
    --num-frames 16 \
    --max-image-size 256 \
    --max-train-steps 1000000 \
    --local-batch-size 2 \
    --lr 1e-4 \
    --ckpt-every 500 \
    --log-every 2 \
    --gradient-checkpointing \
    --attention-mode xformers \
    --mixed-precision bf16



for debug
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch \
    --num_processes 1 \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --ae stabilityai/sd-vae-ft-mse \
    --data-path /root/autodl-tmp/sea.csv \
    --video-folder /root/autodl-tmp/sea \
    --text-encoder-name DeepFloyd/t5-v1_1-xxl \
    --model-max-length 120 \
    --dataset t2v \
    --extras 3 \
    --sample-rate 1 \
    --num-frames 16 \
    --max-image-size 256 \
    --max-train-steps 1000000 \
    --local-batch-size 2 \
    --lr 1e-4 \
    --ckpt-every 500 \
    --log-every 50 \
    --gradient-checkpointing \
    --attention-mode flash \
    --mixed-precision bf16 \
    --num-workers 0 \
    --pretrained PixArt-XL-2-256x256.pth
