import torch
import torch.nn as nn
from streamvc.encoder_decoder import Encoder, Decoder, LearnablePooling
from streamvc.f0 import F0Estimator, F0Whitening
from streamvc.energy import EnergyEstimator


class StreamVC(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = Encoder(scale=64, embedding_dim=64)
        self.speech_encoder = Encoder(scale=32, embedding_dim=64)
        self.speech_pooling = LearnablePooling(dim=64)
        self.decoder = Decoder(scale=40, embedding_dim=64, conditioning_dim=64)
        self.f0_estimator = F0Estimator()
        self.f0_whitening = F0Whitening()
        self.energy_estimator = EnergyEstimator()

    def forward(self, source_speech: torch.Tensor, target_speech: torch.Tensor) -> torch.Tensor:
        content_latent = self.content_encoder(source_speech)
        content_latent = content_latent.detach()
        target_latent = self.speech_encoder(self.speech_pooling(target_speech))
        f0 = self.f0_estimator(self.f0_whitening(source_speech))
        energy = self.energy_estimator(source_speech)
        z = torch.concatenate([content_latent, f0, energy], dim=1)
        output = self.decoder(z, target_latent)
        return output
