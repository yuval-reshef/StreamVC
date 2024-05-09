import torch
import torch.nn as nn
from streamvc.encoder_decoder import Encoder, Decoder
from streamvc.modules import LearnablePooling
from streamvc.f0 import F0Estimator
from streamvc.energy import EnergyEstimator
from streamvc._utils import auto_batching


class StreamVC(nn.Module):
    def __init__(self, sample_rate: int = 16_000, gradient_checkpointing: bool = False):
        super().__init__()
        self.content_encoder = Encoder(scale=64, embedding_dim=64,
                                       gradient_checkpointing=gradient_checkpointing)
        self.speech_encoder = Encoder(scale=32, embedding_dim=64,
                                      gradient_checkpointing=gradient_checkpointing)
        self.speech_pooling = LearnablePooling(embedding_dim=64)
        self.decoder = Decoder(scale=40, embedding_dim=74, conditioning_dim=64,
                               gradient_checkpointing=gradient_checkpointing)
        self.f0_estimator = F0Estimator(sample_rate=sample_rate, frame_length_ms=20,
                                        yin_thresholds=(0.05, 0.1, 1.5), whitening=True)
        self.energy_estimator = EnergyEstimator(
            sample_rate=sample_rate, frame_length_ms=20)

    @auto_batching(('* t', '* t'), '* t')
    def forward(self, source_speech: torch.Tensor, target_speech: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            content_latent = self.content_encoder(source_speech)
            f0 = self.f0_estimator(source_speech)
            energy = self.energy_estimator(source_speech).unsqueeze(dim=-1)
            source_linguistic_features = torch.cat(
                [content_latent, f0, energy], dim=-1)

        target_latent = self.speech_pooling(self.speech_encoder(target_speech))
        return self.decoder(source_linguistic_features, target_latent)
