import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor, gradient_checkpointing: bool = False):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        assert n_layers > 0
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )
        self.gradient_checkpointing = gradient_checkpointing
        self.model = model

    def _run_function(self):
        def custom_forward(*inputs):
            results = []
            x = inputs[0]
            for key, layer in self.model.items():
                x = layer(x)
                results.append(x)
            return results
        return custom_forward

    def forward(self, x: torch.Tensor):
        if self.gradient_checkpointing:
            return checkpoint(self._run_function(), x, use_reentrant=False)
        else:
            return self._run_function()(x)


class Discriminator(nn.Module):
    def __init__(self, n_blocks=3, n_features=16, n_layers=4, downsampling_factor=4,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.model = nn.ModuleList()
        for i in range(n_blocks):
            self.model.append(NLayerDiscriminator(
                n_features, n_layers, downsampling_factor, gradient_checkpointing=gradient_checkpointing
            ))

        self.downsample = nn.AvgPool1d(
            4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        if x.shape[-2] != 1:
            x = x.unsqueeze(-2)
        for D in self.model:
            results.append(D(x))
            x = self.downsample(x)
        return results
