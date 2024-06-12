CUDA_VISIBLE_DEVICES=0 python examples/rec_image.py \
    --ae_path "/storage/dataset/6_10_latent8" \
    --image_path /storage/dataset/image/anytext3m/ocr_data/Art/images/gt_5544.jpg \
    --rec_path rec.jpg \
    --device cuda \
    --short_size 512 \
    --ae CausalVAEModel_D8_4x8x8 \
    --enable_tiling