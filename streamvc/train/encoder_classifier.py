import torch.nn as nn


class EncoderClassifier(nn.Module):
    """A module that wraps an encoder with layer normalization and a linear layer to operate as a classifier."""
    def __init__(self, encoder: nn.Module, in_features: int, out_features: int):
        super(EncoderClassifier, self).__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        return x



