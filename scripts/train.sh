accelerate launch --multi_gpu --num_processes 7 --mixed_precision bf16 opensora/train/train.py \
  --model Latte-XL/122 --dataset sky \
  --ae ucf101_stride4x4x4 \
  --data-path "/remote-home/yeyang/sky_timelapse/sky_train/" --extras 1 \
  --sample-rate 3 --num-frames 128 --max-image-size 256 \
  --max-train-steps 1000000 --local-batch-size 3 --lr 1e-4 \
  --ckpt-every 500 --log-every 50 --gradient-checkpointing --mixed-precision
