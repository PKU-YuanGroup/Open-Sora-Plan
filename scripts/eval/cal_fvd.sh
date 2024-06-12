python opensora/eval/eval_common_metric.py \
    --real_video_dir "/data/xiaogeng_liu/rain/nice_results" \
    --generated_video_dir "/data/xiaogeng_liu/rain/0411_audio_final" \
    --batch_size 5 \
    --crop_size 64 \
    --num_frames 20 \
    --device 'cuda' \
    --metric 'fvd'
