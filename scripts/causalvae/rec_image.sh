CUDA_VISIBLE_DEVICES=0 python opensora/sample/rec_image.py \
    --ae WFVAE2Model_D128_1x16x16 \
    --ae_path "/storage/lcm/WF-VAE_paper/results/WFVAE2_11616128_slim" \
    --image_path /storage/ongoing/12.13/t2i/Open-Sora-Plan/tz_poster/0a9b5a72-3704-41b5-a0fc-ce405ee57741.jpeg \
    --rec_path wfvae11616.jpg \
    --device cuda \
    --short_size 384