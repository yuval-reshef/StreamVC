import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from streamvc.modules import CausalConv1d, CausalConvTranspose1d, FiLM


class Encoder(nn.Module):
    def __init__(self, scale: int, embedding_dim: int, gradient_checkpointing: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            Rearrange('... samples -> ... 1 samples'),
            CausalConv1d(in_channels=1, out_channels=scale, kernel_size=7),
            nn.ELU(),
            EncoderBlock(scale, 2*scale, stride=2,
                         gradient_checkpointing=gradient_checkpointing),
            EncoderBlock(2*scale, 4*scale, stride=4,
                         gradient_checkpointing=gradient_checkpointing),
            EncoderBlock(4*scale, 8*scale, stride=5,
                         gradient_checkpointing=gradient_checkpointing),
            EncoderBlock(8*scale, 16*scale, stride=8,
                         gradient_checkpointing=gradient_checkpointing),
            CausalConv1d(16*scale, embedding_dim, kernel_size=3),
            nn.ELU(),
            Rearrange('... embedding frames -> ... frames embedding')
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, scale: int, embedding_dim: int, conditioning_dim: int, gradient_checkpointing: bool = False):
        super().__init__()
        self.decoder = SequentialWithFiLM(
            Rearrange('... frames embedding -> ... embedding frames'),
            CausalConv1d(embedding_dim, 16*scale, kernel_size=7),
            nn.ELU(),
            DecoderBlock(16*scale, 8*scale, stride=8,
                         gradient_checkpointing=gradient_checkpointing),
            FiLM(8*scale, conditioning_dim),
            DecoderBlock(8*scale, 4*scale, stride=5,
                         gradient_checkpointing=gradient_checkpointing),
            FiLM(4*scale, conditioning_dim),
            DecoderBlock(4*scale, 2*scale, stride=4,
                         gradient_checkpointing=gradient_checkpointing),
            FiLM(2*scale, conditioning_dim),
            DecoderBlock(2*scale, scale, stride=2,
                         gradient_checkpointing=gradient_checkpointing),
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


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(in_channels, dilation=1,
                         gradient_checkpointing=gradient_checkpointing),
            ResidualUnit(in_channels, dilation=3,
                         gradient_checkpointing=gradient_checkpointing),
            ResidualUnit(in_channels, dilation=9,
                         gradient_checkpointing=gradient_checkpointing),
            CausalConv1d(in_channels, out_channels,
                         kernel_size=2*stride, stride=stride),
            nn.ELU()
        )
        self.gradient_checkpointing = gradient_checkpointing

    def _run_function(self):
        def custom_forward(*inputs):
            return self.block(inputs[0])
        return custom_forward

    def forward(self, x: torch.Tensor):
        if self.gradient_checkpointing:
            return checkpoint(self._run_function(), x, use_reentrant=False)
        else:
            return self.block(x)


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            CausalConvTranspose1d(in_channels, out_channels,
                                  kernel_size=2*stride, stride=stride),
            ResidualUnit(out_channels, dilation=1,
                         gradient_checkpointing=gradient_checkpointing),
            ResidualUnit(out_channels, dilation=3,
                         gradient_checkpointing=gradient_checkpointing),
            ResidualUnit(out_channels, dilation=9,
                         gradient_checkpointing=gradient_checkpointing)
        )
        self.gradient_checkpointing = gradient_checkpointing

    def _run_function(self):
        def custom_forward(*inputs):
            return self.block(inputs[0])
        return custom_forward

    def forward(self, x: torch.Tensor):
        if self.gradient_checkpointing:
            return checkpoint(self._run_function(), x, use_reentrant=False)
        else:
            return self.block(x)


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 7, gradient_checkpointing=False, **kwargs):
        super().__init__()
        self.unit = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size,
                         dilation=dilation, **kwargs),
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size=1, **kwargs),
            nn.ELU()
        )
        self.gradient_checkpointing = gradient_checkpointing

    def _run_function(self):
        def custom_forward(*inputs):
            return self.unit(inputs[0]) + inputs[0]
        return custom_forward

    def forward(self, x: torch.Tensor):
        if self.gradient_checkpointing:
            return checkpoint(self._run_function(), x, use_reentrant=False)
        else:
            return self.unit(x) + x
