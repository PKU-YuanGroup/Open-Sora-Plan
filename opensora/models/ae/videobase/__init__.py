from .vqvae import (
    VQVAEConfiguration,
    VQVAEModel,
    VQVAETrainer,
    VQVAEDataset,
)

videovqvae = list(VQVAEModel.DOWNLOADED_VQVAE.keys())
videovae = []

videobase_ae_stride = VQVAEModel.STRIDE

videobase_ae_channel = VQVAEModel.CHANNEL

videobase_ae = {
    "bair_stride4x2x2": VQVAEModel,
    "ucf101_stride4x4x4": VQVAEModel,
    "kinetics_stride4x4x4": VQVAEModel,
    "kinetics_stride2x4x4": VQVAEModel,
}
