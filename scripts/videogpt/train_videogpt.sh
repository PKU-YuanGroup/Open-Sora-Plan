
accelerate launch \
  --config_file scripts/accelerate_configs/ddp_config.yaml \
  opensora/train/train_videogpt.py \
  --do_train \
  --seed 1234 \
  --data_path "/remote-home/yeyang/UCF-101/" \
  --per_device_train_batch_size 1  \
  --gradient_accumulation_steps 1 \
  --learning_rate 7e-4 \
  --weight_decay 0. \
  --max_steps 20000 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --save_strategy steps \
  --save_total_limit 5 \
  --logging_steps 5 \
  --save_steps 1000 \
  --n_codes 2048 \
  --n_hiddens 240 \
  --embedding_dim 4 \
  --n_res_layers 4 \
  --downsample "4,4,4" \
  --resolution 240 \
  --sequence_length 16 \
  --output_dir results/videogpt_488_256_16 \
  --bf16 True \
  --fp16 False \
  --report_to tensorboard \
  --dataloader_num_workers 10
