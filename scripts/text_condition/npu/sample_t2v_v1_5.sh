
export TASK_QUEUE_ENABLE=0
torchrun --nnodes=1 --nproc_per_node 8 --master_port 29513 \
    -m opensora.sample.sample \
    --model_path /home/save_dir/runs/test_t2v_suv_2b_1x384x384_bs16x1x8/checkpoint-14000/model_ema \
    --version v1_5 \
    --num_frames 1 \
    --height 384 \
    --width 384 \
    --cache_dir "../cache_dir" \
    --text_encoder_name_1 "/home/save_dir/pretrained/t5/t5-v1_1-xxl" \
    --text_encoder_name_2 "/home/save_dir/pretrained/clip/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189" \
    --text_prompt examples/sora_refine.txt \
    --ae WFVAEModel_D32_4x8x8 \
    --ae_path "/home/save_dir/lzj/formal_32dim" \
    --save_img_path "./test_sample_v1_5" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --prediction_type "v_prediction" \
    --rescale_betas_zero_snr