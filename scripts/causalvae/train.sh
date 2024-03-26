
accelerate launch \
  --config_file scripts/accelerate_configs/ddp_config.yaml \
  opensora/train/train_causalvae.py \
  scripts/causalvae/config.yaml
  