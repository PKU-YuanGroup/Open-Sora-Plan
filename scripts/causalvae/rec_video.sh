CUDA_VISIBLE_DEVICES=1 python examples/rec_video.py \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/latent8" \
    --video_path /storage/lcm/WF-VAE/testvideo/gm1190263332-337350271.mp4 \
    --rec_path rec_tile.mp4 \
    --device cuda \
    --sample_rate 1 \
    --num_frames 33 \
    --height 256 \
    --width 256 \
    --fps 30 \
    --enable_tiling