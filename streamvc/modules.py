from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import EinMix
from streamvc._utils import auto_batching


class LearnablePooling(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.to_weights = nn.Sequential(
            EinMix('b f e -> b f', weight_shape='e', e=embedding_dim),
            nn.Softmax(dim=-1)
        )

    @auto_batching(('* f e',), '* e')
    def forward(self, x: torch.Tensor):
        weights = self.to_weights(x)
        return torch.einsum('b f e, b f -> b e', x, weights)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1,
                 padding_mode: str = 'zeros', **kwargs):

        assert 'padding' not in kwargs

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            **kwargs
        )

        causal_padding = dilation * (kernel_size - 1) - (stride - 1)

        if padding_mode == 'zeros':
            self.pad = partial(F.pad, pad=(causal_padding, 0),
                               mode='constant', value=0)
        else:
            self.pad = partial(F.pad, pad=(causal_padding, 0),
                               mode=padding_mode)

    def forward(self, x):
        return super().forward(self.pad(x))


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation=1, **kwargs):

        assert 'padding' not in kwargs
        assert 'output_padding' not in kwargs

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=0,
            dilation=dilation,
            **kwargs
        )

        self.causal_trim = dilation * (kernel_size - 1) - (stride - 1)

    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        # we trim the output on the right side
        # see https://github.com/lucidrains/audiolm-pytorch/issues/8
        return out[..., :-self.causal_trim]


class FiLM(nn.Module):
    """
    See paper "FiLM: Visual Reasoning with a General Conditioning Layer"
    """

    def __init__(self, dim: int, conditioning_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(conditioning_dim, dim)
        self.to_beta = nn.Linear(conditioning_dim, dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        gamma = self.to_gamma(condition).unsqueeze(dim=-1)
        beta = self.to_beta(condition).unsqueeze(dim=-1)
        return x * gamma + beta
