CUDA_VISIBLE_DEVICES=0 python examples/rec_imvi_vae.py \
    --ae_path "/dxyl_data02/CausalVAEModel_4x8x8/" \
    --video_path /dxyl_data02/datasets/mixkit/Airplane/mixkit-sun-rays-over-forest-treetops-515.mp4 \
    --rec_path rec_488_129f_512.mp4 \
    --device cuda \
    --sample_rate 1 \
    --num_frames 513 \
    --resolution 256 \
    --crop_size 256 \
    --ae CausalVAEModel_4x8x8 