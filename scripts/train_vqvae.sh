export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29409 ../opensora/models/ae/videobase/vqvae/videogpt/scripts/train_vqvae.py \
    --data_path='../datasets/bair.hdf5') \
    --sequence_length=16 \
    --resolution=64 \
    --batch_size=32 \
    --num_workers=8