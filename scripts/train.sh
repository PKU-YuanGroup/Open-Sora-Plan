accelerate launch --config_file scripts/accelerate_configs/default_config.yaml \
  opensora/train/train.py \
  --model Latte-XL/122 --dataset ucf101 \
  --ae ucf101_stride4x4x4 \
  --data-path "/data03/dragon_proj/Open-Sora-Plan/UCF-101/train/" --extras 1 \
  --sample-rate 3 --num-frames 128 --max-image-size 256 \
  --max-train-steps 1000000 --local-batch-size 3 --lr 1e-4 \
  --ckpt-every 500 --log-every 50 --gradient-checkpointing --mixed-precision