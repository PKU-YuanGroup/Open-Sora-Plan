accelerate launch \
    --num_processes 1 \
    --main_process_port 29502 \
    opensora/sample/sample_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --prompt "A woman" \
    --ae stabilityai/sd-vae-ft-mse \
    --ckpt t2v-f16s3-256-imgvae188-bf16-ckpt-xformers/checkpoint-100000 \
    --extras 3 \
    --fps 10 \
    --num-frames 16 \
    --image-size 256 \
    --num-sampling-steps 250 \
    --attention-mode xformers \
    --mixed-precision bf16
