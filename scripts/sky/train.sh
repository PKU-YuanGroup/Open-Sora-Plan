

CUDA_VISIBLE_DEVICES=5,6 accelerate launch --multi_gpu --num_processes 2 --main_process_port 29503 --mixed_precision bf16 opensora/train/train.py \
  --model Latte-XL/122 --dataset sky \
  --ae stabilityai/sd-vae-ft-mse \
  --data-path "/remote-home/yeyang/sky_timelapse/sky_train/" --extras 1 \
  --sample-rate 1 --num-frames 16 --max-image-size 256 \
  --max-train-steps 1000000 --local-batch-size 5 --lr 1e-4 --num-workers 10 \
  --ckpt-every 500 --log-every 50 --gradient-checkpointing --attention_mode flash