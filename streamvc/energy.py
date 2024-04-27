import torch
import torch.nn as nn


class EnergyEstimator(nn.Module):
    """
    Calculates the energy estimator for each frame measured by sample variance.
    """
    def __init__(self, samples_per_frame: int):
        super().__init__()
        self.samples_per_frame = samples_per_frame

    def reshape_to_frames(self, tensor):
        """
        Reshapes the tensor from dimension [..., x] to [..., x / samples_per_frame, samples_per_frame].
        :param tensor: Input tensor to reshape.
        :return: Reshaped tensor with the specified frame structure.
        """
        original_shape = tensor.size()
        new_shape = (*original_shape[:-1], original_shape[-1] // self.samples_per_frame, self.samples_per_frame)
        reshaped_tensor = tensor.view(*new_shape)
        return reshaped_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) % self.samples_per_frame == 0
        reshaped_tensor = self.reshape_to_frames(x)
        return torch.var(reshaped_tensor, dim=-1)
