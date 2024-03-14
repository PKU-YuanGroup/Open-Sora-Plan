
accelerate launch \
    --config_file scripts/accelerate_configs/ddp_config.yaml \
    opensora/train/train.py \
    --model Latte-XL/122 \
    --dataset ucf101 \
    --num-classes 101 \
    --ae stabilityai/sd-vae-ft-mse \
    --data-path /remote-home/yeyang/UCF-101 \
    --extras 2 \
    --sample-rate 3 \
    --num-frames 16 \
    --max-image-size 256 \
    --max-train-steps 1000000 \
    --local-batch-size 5 \
    --lr 1e-4 \
    --ckpt-every 500 \
    --log-every 50 \
    --gradient-checkpointing \
    --attention-mode flash \
    --mixed-precision bf16

#for debug
#accelerate launch \
#    --config_file scripts/accelerate_configs/ddp_config.yaml \
#    opensora/train/train.py \
#    --model Latte-XL/122 \
#    --dataset ucf101 \
#    --num-classes 101 \
#    --ae stabilityai/sd-vae-ft-mse \
#    --data-path /remote-home/yeyang/UCF-101 \
#    --extras 2 \
#    --sample-rate 3 \
#    --num-frames 16 \
#    --max-image-size 256 \
#    --max-train-steps 1000000 \
#    --local-batch-size 5 \
#    --lr 1e-4 \
#    --ckpt-every 500 \
#    --log-every 50 \
#    --gradient-checkpointing \
#    --attention-mode flash \
#    --mixed-precision bf16 \
#    --num-workers 0