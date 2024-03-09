

accelerate launch --multi_gpu --num_processes 7  --mixed_precision bf16 opensora/train/train_with_feature.py \
  --model Latte-XL/122 --dataset landscope_feature \
  --ae stabilityai/sd-vae-ft-mse \
  --data-path "/remote-home/yeyang/train_movie/*/*256.npy" --extras 1 \
  --sample-rate 3 --num-frames 128 --max-image-size 256 \
  --max-train-steps 1000000 --local-batch-size 5 --lr 1e-4 \
  --ckpt-every 500 --log-every 50  --gradient-checkpointing