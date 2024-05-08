import torch
import torch.nn as nn


class EnergyEstimator(nn.Module):
    """
    Calculates the energy estimator for each frame measured by sample variance.
    """
    def __init__(self, sample_rate: int, frame_length_ms: int):
        super().__init__()
        self.samples_per_frame = int(sample_rate // (1 / frame_length_ms * 1000))

    def reshape_to_frames(self, tensor):
        """
        Reshapes the tensor from dimension [..., x] to [..., x // samples_per_frame, samples_per_frame].
        :param tensor: Input tensor to reshape.
        :return: Reshaped tensor with the specified frame structure.
        """
        # Remove any samples at the end that don't fill a whole frame.
        tensor = tensor[..., :tensor.shape[-1] // self.samples_per_frame * self.samples_per_frame]
        # Reshape the tensor.
        original_shape = tensor.size()
        new_shape = (*original_shape[:-1], original_shape[-1] // self.samples_per_frame, self.samples_per_frame)
        return tensor.view(*new_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped_tensor = self.reshape_to_frames(x)
        return torch.var(reshaped_tensor, dim=-1)
