export MASTER_PORT=12359
export MASTER_ADDR='localhost'
export TOKENIZERS_PARALLELISM=false
MODEL_PATH=inpaint_93x480p_bs4x8x1_lr1e-5_snr5_noioff0.02_new_mask_onlysucai
# export HF_DATASETS_OFFLINE=1 
# export TRANSFORMERS_OFFLINE=1

torchrun --nproc_per_node=8 opensora/sample/sample_inpaint_all_in_one.py \
    --model_path /storage/gyy/hw/Open-Sora-Plan/runs/$MODEL_PATH \
    --model_type 'inpaint_only' \
    --num_frames 93 \
    --height 720 \
    --width 1280 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --save_img_path "./samples/$MODEL_PATH" \
    --fps 24 \
    --guidance_scale 5.0 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --validation_dir "./validation_dir" \
    --pretrained_transformer_model_path /storage/ongoing/new/Open-Sora-Plan/bs32x8x1_93x720p_lr2e-5_snr5_ema999_opensora122_rope_fp32_mt5xxl_sucai_aes5.5_speed1.5/checkpoint-400/model_ema \
    --pretrained_vipnet_path /storage/gyy/hw/Open-Sora-Plan/runs/videoip_3d_480p_f93_bs1x16_lr1e-5_snrgamma5_0_noiseoffset0_02_dino518_ema0_999/checkpoint-3000/model \
    --image_encoder_name vit_giant_patch14_reg4_dinov2.lvd142m \
    --image_encoder_path /storage/cache_dir/hub/models--timm--vit_giant_patch14_reg4_dinov2.lvd142m/snapshots/a2208b21b069f6b2e45999870fcce4b7e43d1a2c/model.safetensors \
    --max_sample_num 8 \
    --seed 42