torchrun  --nproc_per_node=8 --master_port=29501 opensora/train/train.py \
  --model Latte-XL/122 --dataset sky \
  --ae stabilityai/sd-vae-ft-mse \
  --data-path /remote-home/yeyang/sky_timelapse/sky_train/ --extras 1 \
  --sample-rate 3 --num-frames 16 --max-image-size 256 \
  --epochs 14000 --global-batch-size 40 --lr 1e-4 \
  --ckpt-every 1000 --log-every 50
