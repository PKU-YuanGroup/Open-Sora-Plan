

accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 opensora/train/train.py \
  --model Latte-XL/122 --dataset sky \
  --ae stabilityai/sd-vae-ft-mse \
  --data-path "/remote-home/yeyang/sky_timelapse/sky_train/" --extras 1 \
  --sample-rate 1 --num-frames 128 --max-image-size 256 \
  --max-train-steps 1000000 --local-batch-size 2 --lr 1e-4 --num-workers 10 \
  --ckpt-every 500 --log-every 50 --gradient-checkpointing --attention_mode flash