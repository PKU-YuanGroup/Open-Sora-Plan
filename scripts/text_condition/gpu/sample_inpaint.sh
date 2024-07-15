export MASTER_PORT=12359
export MASTER_ADDR='localhost'

torchrun --nproc_per_node=8 opensora/sample/sample_inpaint.py \
    --model_path /storage/gyy/hw/Open-Sora-Plan/inpaint_93x480p_bs4x8x1_lr1e-5_snr5_noioff0.02_new_mask_onlysucai \
    --num_frames 93 \
    --height 480 \
    --width 640 \
    --cache_dir "../cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "/storage/dataset/test140k" \
    --save_img_path "/storage/gyy/hw/Open-Sora-Plan/sample_inpaint_93x480p_bs4x8x1_lr1e-5_snr5_noioff0.02_new_mask_onlysucai" \
    --fps 24 \
    --guidance_scale 5.0 \
    --num_sampling_steps 50 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method PNDM \
    --model_3d \
    --validation_dir "./validation_dir" \
    --seed 42