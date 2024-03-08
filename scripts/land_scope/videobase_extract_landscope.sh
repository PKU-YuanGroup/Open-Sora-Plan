torchrun  --nproc_per_node=7 opensora/utils/videobase_extract_video_feature.py \
  --data-path "/remote-home/yeyang/train_movie/*/*.mp4" \
  --ae ucf101_stride4x4x4 --features-name "landscope-s256-videogpt-f16" --image-size 256 \
  --n-frame-per-sample 16 --sample-rate 3 --num-workers 4