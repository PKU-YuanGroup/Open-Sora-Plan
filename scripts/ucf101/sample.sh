accelerate launch \
  --num_processes 1 \
  --main_process_port 29502 \
  opensora/sample/sample.py \
  --model Latte-XL/122 \
  --ae stabilityai/sd-vae-ft-mse \
  --ckpt ucf101-f16s3-256-imgvae188-bf16-ckpt-flash/checkpoint-100000 \
  --extras 2 \
  --fps 10 \
  --num-frames 16 \
  --image-size 256 \
  --num-sampling-steps 250 \
  --attention-mode flash \
  --mixed-precision bf16
