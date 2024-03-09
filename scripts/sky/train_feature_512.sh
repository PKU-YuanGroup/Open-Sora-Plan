

accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 opensora/train/train_with_feature.py \
  --model Latte-XL/122 --dataset landscope_feature \
  --ae stabilityai/sd-vae-ft-mse \
  --data-path "/remote-home/yeyang/sky_timelapse/sky_train/" --extras 1 \
  --sample-rate 3 --num-frames 64 --max-image-size 512 \
  --max-train-steps 1000000 --local-batch-size 5 --lr 1e-4 \
  --ckpt-every 500 --log-every 50 --gradient-checkpointing --attention_mode flash