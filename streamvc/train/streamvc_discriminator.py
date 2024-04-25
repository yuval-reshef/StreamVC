import torch.nn as nn
from streamvc.model import StreamVC


class StreamVCDiscriminator(nn.Module):
    """A module that contains a StreamVC model followed by a discriminator for training purposes."""
    def __init__(self, streamvc: StreamVC):
        super(StreamVCDiscriminator, self).__init__()
        pass


    def forward(self, x):
        # TODO add the discriminator.
        return x

