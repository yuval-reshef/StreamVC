import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from streamvc.modules import CausalConv1d, CausalConvTranspose1d, FiLM, Residual


class Encoder(nn.Module):
    def __init__(self, scale: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            Rearrange('... samples -> ... 1 samples'),
            CausalConv1d(in_channels=1, out_channels=scale, kernel_size=7),
            nn.ELU(),
            EncoderBlock(scale, 2*scale, stride=2),
            EncoderBlock(2*scale, 4*scale, stride=4),
            EncoderBlock(4*scale, 8*scale, stride=5),
            EncoderBlock(8*scale, 16*scale, stride=8),
            CausalConv1d(16*scale, embedding_dim, kernel_size=3),
            nn.ELU(),
            Rearrange('... embedding frames -> ... frames embedding')
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, scale: int, embedding_dim: int, conditioning_dim: int):
        super().__init__()
        self.decoder = SequentialWithFiLM(
            Rearrange('... frames embedding -> ... embedding frames'),
            CausalConv1d(embedding_dim, 16*scale, kernel_size=7),
            nn.ELU(),
            DecoderBlock(16*scale, 8*scale, stride=8),
            FiLM(8*scale, conditioning_dim),
            DecoderBlock(8*scale, 4*scale, stride=5),
            FiLM(4*scale, conditioning_dim),
            DecoderBlock(4*scale, 2*scale, stride=4),
            FiLM(2*scale, conditioning_dim),
            DecoderBlock(2*scale, scale, stride=2),
            FiLM(scale, conditioning_dim),
            CausalConv1d(scale, 1, kernel_size=7),
            Rearrange('... 1 samples -> ... samples')
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        return self.decoder(x, condition)


class SequentialWithFiLM(nn.Sequential):
    def forward(self, input, condition):
        for module in self:
            if isinstance(module, FiLM):
                input = module(input, condition)
            else:
                input = module(input)
        return input


def EncoderBlock(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    return nn.Sequential(
        ResidualUnit(in_channels, dilation=1),
        ResidualUnit(in_channels, dilation=3),
        ResidualUnit(in_channels, dilation=9),
        CausalConv1d(in_channels, out_channels,
                     kernel_size=2*stride, stride=stride),
        nn.ELU()
    )


def DecoderBlock(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    return nn.Sequential(
        CausalConvTranspose1d(in_channels, out_channels,
                              kernel_size=2*stride, stride=stride),
        ResidualUnit(out_channels, dilation=1),
        ResidualUnit(out_channels, dilation=3),
        ResidualUnit(out_channels, dilation=9),
    )


def ResidualUnit(channels: int, dilation: int, kernel_size: int = 7, **kwargs):
    return Residual(
        nn.Sequential(
            CausalConv1d(channels, channels, kernel_size,
                         dilation=dilation, **kwargs),
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size=1, **kwargs),
            nn.ELU()
        )
    )
