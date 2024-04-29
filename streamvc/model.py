import torch
import torch.nn as nn
from einops import pack
from streamvc.encoder_decoder import Encoder, Decoder
from streamvc.modules import LearnablePooling
from streamvc.f0 import F0Estimator
from streamvc.energy import EnergyEstimator
from streamvc._utils import auto_batching


class StreamVC(nn.Module):
    def __init__(self, samples_per_frame: int, sample_rate: int, gradient_checkpointing: bool = False):
        super().__init__()
        self.content_encoder = Encoder(scale=64, embedding_dim=64,
                                       gradient_checkpointing=gradient_checkpointing)
        self.speech_encoder = Encoder(scale=32, embedding_dim=64,
                                      gradient_checkpointing=gradient_checkpointing)
        self.speech_pooling = LearnablePooling(embedding_dim=64)
        self.decoder = Decoder(scale=40, embedding_dim=64, conditioning_dim=64,
                               gradient_checkpointing=gradient_checkpointing)
        self.f0_estimator = F0Estimator(
            sample_rate, samples_per_frame, whitening=True)
        self.energy_estimator = EnergyEstimator(samples_per_frame)

    @auto_batching(('* t', '* t'), '* t')
    def forward(self, source_speech: torch.Tensor, target_speech: torch.Tensor) -> torch.Tensor:
        content_latent = self.content_encoder(source_speech)
        content_latent = content_latent.detach()
        target_latent = self.speech_pooling(self.speech_encoder(target_speech))
        f0 = self.f0_estimator(source_speech)
        energy = self.energy_estimator(source_speech)
        z = pack([content_latent, f0, energy], 'b f *')
        return self.decoder(z, target_latent)
