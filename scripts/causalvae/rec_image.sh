CUDA_VISIBLE_DEVICES=0 python examples/rec_image.py \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/storage/lcm/WF-VAE/results/latent8" \
    --image_path /storage/dataset/image/anytext3m/ocr_data/Art/images/gt_5544.jpg \
    --rec_path rec_.jpg \
    --device cuda \
    --short_size 512 