torchrun  --nproc_per_node=7 opensora/utils/imagebase_extract_video_feature.py \
  --data-path "/remote-home/yeyang/train_movie/*/*.mp4" \
  --ae stabilityai/sd-vae-ft-mse --features-name "landscope-256" --image-size 256
