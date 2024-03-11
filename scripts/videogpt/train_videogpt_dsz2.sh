export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=4

accelerate launch \
  --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
  opensora/train/train_videogpt.py \
  --do_train \
  --seed 1234 \
  --data_path datasets/UCF-101 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps $ACCELERATE_GRADIENT_ACCUMULATION_STEPS \
  --learning_rate 1e-5 \
  --weight_decay 0. \
  --num_train_epochs 2 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --save_strategy steps \
  --save_total_limit 5 \
  --logging_steps 5 \
  --save_steps 1000 \
  --n_codes 2048 \
  --n_hiddens 512 \
  --embedding_dim 4 \
  --n_res_layers 4 \
  --downsample "4,8,8" \
  --resolution 128 \
  --sequence_length 16 \
  --bf16 False \
  --fp16 True \
  --output_dir results/videogpt \
  --report_to tensorboard
