
accelerate launch \
  --num_processes 1 \
  --main_process_port 29502 \
  opensora/sample/sample.py \
  --model Latte-XL/122 \
  --ae CausalVQVAEModel \
  --ckpt sky-f17s3-128-causalvideovae488-bf16-ckpt-flash-log/checkpoint-45500 \
  --fps 10 \
  --num_frames 17 \
  --image_size 128 \
  --num_sampling_steps 250 \
  --attention_mode flash \
  --mixed_precision bf16 \
  --num_sample 10