export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1

accelerate launch \
  --config_file scripts/accelerate_configs/ddp_config.yaml \
  opensora/train/train_ae.py \
  --model_name causalvqvae \
  --do_train \
  --seed 1234 \
  --video_data_path datasets/kinetics-400 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps $ACCELERATE_GRADIENT_ACCUMULATION_STEPS \
  --learning_rate 7e-4 \
  --weight_decay 0. \
  --max_steps 100000 \
  --embedding_dim 4 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --save_strategy steps \
  --save_total_limit 5 \
  --logging_steps 5 \
  --save_steps 1000 \
  --n_codes 8192 \
  --n_hiddens 256 \
  --n_res_layers 4 \
  --time_downsample 4 \
  --spatial_downsample 8 \
  --bf16 False \
  --fp16 True \
  --resolution 256 \
  --sequence_length 17 \
  --output_dir results/casualvqvae_kinetics \
  --dataloader_num_workers 8 \
  --report_to tensorboard