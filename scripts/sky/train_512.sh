
accelerate launch \
  --config_file scripts/accelerate_configs/ddp_config.yaml \
  opensora/train/train.py \
  --model Latte-XL/122 --dataset sky \
  --ae stabilityai/sd-vae-ft-mse \
  --data-path "/remote-home/yeyang/sky_timelapse/sky_train/" --extras 1 \
  --sample-rate 3 --num-frames 16 --max-image-size 512 \
  --max-train-steps 1000000 --local-batch-size 2 --lr 1e-4 \
  --ckpt-every 500 --log-every 50 --gradient-checkpointing --attention_mode flash  --mixed_precision bf16
