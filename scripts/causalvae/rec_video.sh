CUDA_VISIBLE_DEVICES=1 python examples/rec_video.py \
<<<<<<< HEAD
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "WFVAE_DISTILL_FORMAL" \
    --video_path 134445.mp4 \
=======
    --ae CausalVAEModel_D8_4x8x8 \
    --ae_path "/storage/dataset/new488dim8/last" \
    --video_path /storage/dataset/mixkit-train-passing-the-rails-4462_resize1080p.mp4 \
>>>>>>> 7f41973 (fix resume)
    --rec_path rec.mp4 \
    --device cuda \
    --sample_rate 1 \
    --num_frames 65 \
<<<<<<< HEAD
    --height 480 \
    --width 640 \
    --fps 30
=======
    --height 512 \
    --width 512 \
    --fps 30 \
    --enable_tiling \
    --tile_overlap_factor 0.0
>>>>>>> 7f41973 (fix resume)
