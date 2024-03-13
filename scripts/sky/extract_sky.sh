HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 torchrun  --nproc_per_node=7 opensora/utils/imagebase_extract_image_feature.py \
  --data-path "/remote-home/yeyang/sky_timelapse/sky_train/*/*/*.jpg" \
  --ae stabilityai/sd-vae-ft-mse --features-name "sky-256" --image-size 256 \
  --num-workers 10
