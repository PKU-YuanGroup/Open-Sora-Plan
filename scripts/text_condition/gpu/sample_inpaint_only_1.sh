export MASTER_ADDR='localhost'
export TOKENIZERS_PARALLELISM=false
MODEL_PATH=inpaint_only_480p_f93_bs4x8x1_lr1e-5_snrgamma5_0_noiseoffset0_02_ema0_999_old_script
# export HF_DATASETS_OFFLINE=1 
# export TRANSFORMERS_OFFLINE=1

torchrun --nproc_per_node=8 --master_port=29501 opensora/sample/sample_inpaint.py \
    --model_path /storage/gyy/hw/Open-Sora-Plan/runs/$MODEL_PATH \
    --num_frames 93 \
    --height 480 \
    --width 640 \
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
    --sample_method PNDM \
    --validation_dir "./validation_dir" \
    --seed 42