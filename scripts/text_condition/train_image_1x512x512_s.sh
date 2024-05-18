export WANDB_KEY="953e958793b218efb850fa194e85843e2c3bd88b"
export ENTITY="linbin"
export PROJECT="256_lr1e-4_bs64_1node_s"
#export HF_ENDPOINT="https://hf-mirror.com"
#export HF_DATASETS_OFFLINE=1 
#export TRANSFORMERS_OFFLINE=1
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v.py \
    --model OpenSoraT2V-S/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/remote-home1/yeyang/CausalVAEModel_4x8x8/" \
    --image_data "scripts/train_data/image_data.txt" \
    --sample_rate 1 \
    --num_frames 1 \
    --max_image_size 256 \
    --attention_mode xformers \
    --train_batch_size=64 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-04 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="256_lr1e-4_bs64_1node_s" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 0 \
    --enable_tiling \
    --enable_tracker \
    --resume_from_checkpoint "latest"
