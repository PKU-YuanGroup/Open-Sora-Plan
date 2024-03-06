torchrun  --nproc_per_node=1 --master_port=29501 opensora/train/train.py \
  --model DiT-XL/122 --dataset ucf101\
  --ae ucf101_stride4x4x4 \
  --data-path /remote-home/yeyang/UCF-101 --num-classes 101 \
  --sample-rate 2 --num-frames 8 --max-image-size 128 --clip-grad-norm 1 \
  --epochs 14000 --global-batch-size 2 --lr 1e-4 \
  --ckpt-every 1000 --log-every 50