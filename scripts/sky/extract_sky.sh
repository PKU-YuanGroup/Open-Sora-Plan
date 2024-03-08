torchrun  --nproc_per_node=8 opensora/utils/imagebase_extract_image_feature.py \
  --data-path "/remote-home/yeyang/sky_timelapse/sky_train" \
  --ae stabilityai/sd-vae-ft-mse --features-name "sky-512" --image-size 512 \
  --num-workers 4
