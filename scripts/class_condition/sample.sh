accelerate launch \
  --num_processes 1 \
  --main_process_port 29502 \
  opensora/sample/sample.py \
  --model Latte-XL/122 \
  --ae stabilityai/sd-vae-ft-mse \
  --ckpt ucf101-f16s3-128-imgvae188-bf16-ckpt-flash/checkpoint-98500 \
  --train_classcondition \
  --num_classes 101 \
  --fps 10 \
  --num_frames 16 \
  --image_size 128 \
  --num_sampling_steps 500 \
  --attention_mode flash \
  --mixed_precision bf16
