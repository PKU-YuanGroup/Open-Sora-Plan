

accelerate launch --multi_gpu --num_processes 7 opensora/train/train_with_feature.py \
  --model Latte-XL/122 --dataset landscope_video_feature \
  --ae ucf101_stride4x4x4 \
  --data-path "/remote-home/yeyang/train_movie/*/*s256-videogpt-f16*.npy" --extras 1 \
  --sample-rate 3 --num-frames 128 --max-image-size 256 \
  --max-train-steps 1000000 --local-batch-size 16 --lr 1e-4 \
  --ckpt-every 500 --log-every 50