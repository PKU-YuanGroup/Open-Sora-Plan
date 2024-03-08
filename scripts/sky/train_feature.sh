

accelerate launch  --num_processes 8  opensora/train/train_with_feature.py \
  --model Latte-XL/122 --dataset sky_feature \
  --ae stabilityai/sd-vae-ft-mse \
  --data-path "/remote-home/yeyang/sky_timelapse/sky_train/" --extras 1 \
  --sample-rate 3 --num-frames 128 --max-image-size 256 \
  --max-train-steps 1000000 --local-batch-size 5 --lr 1e-4 \
  --ckpt-every 500 --log-every 50 --num-workers 10  --gradient-checkpointing