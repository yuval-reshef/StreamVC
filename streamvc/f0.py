import torch
import torch.nn as nn


class F0Estimator(nn.Module):
    def __init__(self, whitening: bool = False):
        super().__init__()
        self.whitening = whitening

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
