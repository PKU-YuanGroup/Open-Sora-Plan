accelerate launch \
  --num_processes 1 \
  --main_process_port 29502 \
  opensora/sample/sample.py \
  --model Latte-XL/122 \
  --ae stabilityai/sd-vae-ft-mse \
  --ckpt results/015-Latte-XL-122-sd-vae-ft-mse-F16S3-ucf101-FLASH-Gc-BF16-256/checkpoints/0009000.pt \
  --extras 2 \
  --fps 10 \
  --num-frames 16 \
  --image-size 256 \
  --num-sampling-steps 250 \
  --attention-mode flash \
  --mixed-precision bf16 \
  --cfg-scale 4.0

