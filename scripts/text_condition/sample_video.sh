export ASCEND_RT_VISIBLE_DEVICES=0
python opensora/sample/sample_t2v.py \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt "The video captures the majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. The camera angle provides a bird's eye view of the waterfall, allowing viewers to appreciate the full height and grandeur of the waterfall. The video is a stunning representation of nature's power and beauty." \
    --ae CausalVAEModel_4x8x8 \
    --ckpt checkpoint-6500 \
    --fps 24 \
    --num_frames 65 \
    --image_size 512 \
    --num_sampling_steps 100 \
    --enable_tiling
