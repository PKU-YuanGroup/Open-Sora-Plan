python opensora/sample/sample.py \
  --model Latte-XL/122 --ae stabilityai/sd-vae-ft-mse \
  --ckpt results/011-Latte-XL-122-F128S3-landscope_feature-Gc-Amp/checkpoints/0005000.pt --extras 1 \
  --fps 10 --num-frames 128 --image-size 256 --num-sampling-steps 250

