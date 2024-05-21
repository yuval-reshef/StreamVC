from contextlib import contextmanager
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

    @contextmanager
    def streaming(self, target_speech: torch.Tensor):
        gradient_checkpointing = False
        try:
            with torch.no_grad():
                target_latent = self.speech_pooling(
                    self.speech_encoder(target_speech))
            for module in self.modules():
                if hasattr(module, 'streaming_mode'):
                    module.streaming_mode = True
                if hasattr(module, 'init_streaming_buffer'):
                    module.init_streaming_buffer()
                if hasattr(module, 'gradient_checkpointing'):
                    if module.gradient_checkpointing:
                        gradient_checkpointing = True
                    module.gradient_checkpointing = False

            yield _StreamingStreamVC(self, target_latent)

        finally:
            for module in self.modules():
                if hasattr(module, 'streaming_mode'):
                    module.streaming_mode = False
                if hasattr(module, 'remove_streaming_buffer'):
                    module.remove_streaming_buffer()
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = gradient_checkpointing


class _StreamingStreamVC():
    def __init__(self, model: StreamVC, target_latent: torch.Tensor):
        self.model = model
        self.target_latent = target_latent

    @torch.no_grad()
    def forward(self, source_speech_chunck: torch.Tensor):
        content_latent = self.model.content_encoder(source_speech_chunck)
        f0 = self.model.f0_estimator(source_speech_chunck)
        energy = self.model.energy_estimator(
            source_speech_chunck).unsqueeze(dim=-1)
        source_linguistic_features = torch.cat(
            [content_latent, f0, energy], dim=-1)
        return self.model.decoder(source_linguistic_features, self.target_latent)
