export WANDB_KEY=""
export ENTITY=""
export PROJECT="512_2node_bs2"
accelerate launch \
    --config_file scripts/accelerate_configs/multi_node_example.yaml \
    opensora/train/train_t2v.py \
    --model OpenSoraT2V-XL/222 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir "./cache_dir" \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/dxyl_data02/CausalVAEModel_4x8x8/" \
    --video_data "scripts/train_data/video_data.txt" \
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
    --output_dir="512_2node_bs2" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 4 \
    --enable_tiling \
    --use_img_from_vid \
    --pretrained /dxyl_data02/PixArt-Sigma-XL-2-512-MS.safetensors \
    --enable_tracker
