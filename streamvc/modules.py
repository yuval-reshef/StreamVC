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
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.causal_padding = max(
            0, dilation * (kernel_size - 1) - (stride - 1))
        self.padding_mode = padding_mode

        self.register_buffer('streaming_buffer',
                             torch.tensor([]), persistent=False)
        self.streaming_mode = False

    def _pad(self, x):
        if self.padding_mode == 'zeros':
            return F.pad(x, pad=(self.causal_padding, 0),
                         mode='constant', value=0)
        return F.pad(x, pad=(self.causal_padding, 0),
                     mode=self.padding_mode)

    def forward(self, x):
        if self.streaming_mode:
            return self.streaming_forward(x)

        return super().forward(self._pad(x))

    def init_streaming_buffer(self):
        self.streaming_buffer = torch.zeros(
            self.in_channels, self.causal_padding, device=next(iter(self.params())).device)

    def remove_streaming_buffer(self):
        self.streaming_buffer = torch.tensor(
            [], device=next(iter(self.params())).device)

    def streaming_forward(self, x):
        if self.streaming_buffer.numel() == 0:
            full_input = x
        else:
            full_input = torch.cat([self.streaming_buffer, x], dim=-1)

        num_samples = full_input.shape[-1]
        kernel_reception_field = self.dilation * (self.kernel_size - 1) + 1
        num_strides = (num_samples - kernel_reception_field) // self.stride + 1
        num_elements_for_forward = kernel_reception_field + \
            (num_strides - 1) * self.stride
        ready_input = full_input[..., :num_elements_for_forward]
        new_buffer_size = num_samples - num_strides * self.stride
        self.streaming_buffer = full_input[..., -new_buffer_size:]
        return super().forward(ready_input)


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

        causal_trim = max(0, dilation * (kernel_size - 1) - (stride - 1))

        # we trim the output on the right side
        # see https://github.com/lucidrains/audiolm-pytorch/issues/8
        if causal_trim > 0:
            self.trim = lambda x: x[..., :-causal_trim]
        else:
            self.trim = lambda x: x

    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        return self.trim(out)


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
