
accelerate launch \
    --config_file scripts/accelerate_configs/ddp_config.yaml \
    opensora/train/train.py \
    --model Latte-XL/122 \
    --dataset sky \
    --ae stabilityai/sd-vae-ft-mse \
    --data-path /remote-home/yeyang/sky_timelapse/sky_train/ \
    --extras 1 \
    --sample-rate 1 \
    --num-frames 128 \
    --max-image-size 256 \
    --max-train-steps 1000000 \
    --local-batch-size 2 \
    --lr 1e-4 \
    --ckpt-every 500 \
    --log-every 50 \
    --gradient-checkpointing \
    --attention-mode flash \
    --mixed-precision bf16

#for debug
#accelerate launch \
#    --num_processes 1 \
#    opensora/train/train.py \
#    --model Latte-XL/122 \
#    --dataset sky \
#    --ae checkpoint-14000 \
#    --data-path /remote-home/yeyang/sky_timelapse/sky_train/ \
#    --extras 1 \
#    --sample-rate 2 \
#    --num-frames 16 \
#    --max-image-size 256 \
#    --max-train-steps 1000000 \
#    --local-batch-size 2 \
#    --lr 1e-4 \
#    --ckpt-every 500 \
#    --log-every 50 \
#    --gradient-checkpointing \
#    --attention-mode flash \
#    --mixed-precision bf16 \
#    --num-workers 0 \
#    --pretrained PixArt-XL-2-256x256.pth