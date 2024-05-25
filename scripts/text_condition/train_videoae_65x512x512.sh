export WANDB_KEY=""
# export WANDB_MODE="offline"
export ENTITY="linbin"
export PROJECT="512_10node_bs2_lr2e_4img_"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
# export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ibp25s0
export NCCL_P2P_DISABLE=1
export PDSH_PATH_TYPE=ssh
accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example.yaml \
    opensora/train/train_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/dxyl_data02/CausalVAEModel_4x8x8/" \
    --video_data "scripts/train_data/video_data.txt" \
    --image_data "scripts/train_data/image_data.txt" \
    --sample_rate 1 \
    --num_frames 65 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=2 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="512_10node_bs2_lr2e-5_4img" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 4 \
    --enable_tiling \
    --pretrained t2v.pt \
    --enable_tracker \
    --resume_from_checkpoint "latest" \
    --multi_scale
