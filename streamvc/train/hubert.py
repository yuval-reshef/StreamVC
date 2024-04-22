import torch.nn as nn


class HubertEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, in_features: int, out_features: int):
        super(HubertEncoder, self).__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x



