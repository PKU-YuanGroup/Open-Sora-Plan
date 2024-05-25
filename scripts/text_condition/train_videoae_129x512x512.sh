export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
# export WANDB_MODE="offline"
export ENTITY="linbin"
export PROJECT="129f_512_10node_bs1_lr2e-5_4img"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
# export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ibp25s0
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
    --video_data "scripts/train_data/video_data_513.txt" \
    --image_data "scripts/train_data/image_data.txt" \
    --sample_rate 1 \
    --num_frames 129 \
    --max_image_size 512 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=2 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="129f_512_10node_bs1_lr2e-5_4img" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 4 \
    --enable_tiling \
    --pretrained 512_10node_bs2_lr2e-5_4img/checkpoint-94000/model/diffusion_pytorch_model.safetensors \
    --enable_tracker \
    --resume_from_checkpoint "latest" 