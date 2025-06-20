import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from torch.utils.checkpoint import checkpoint


def masked_mean_from_ratios(tensor: torch.Tensor, mask_ratio: torch.Tensor):
    assert len(tensor.shape) == 3
    assert len(mask_ratio.shape) == 1
    assert mask_ratio.shape[0] == tensor.shape[0]

    _, _, samples = tensor.shape
    original_lengths = torch.floor(mask_ratio * samples).to(torch.int32)
    num_original_samples = original_lengths.sum()
    mask = (
        torch.arange(samples, device=tensor.device) < original_lengths.unsqueeze(1)
    ).unsqueeze(1)
    return torch.where(mask, tensor, 0).sum() / num_original_samples


class DiscriminatorLoss(nn.Module):
    def forward(
        self,
        real: list[list[torch.Tensor]],
        fake: list[list[torch.Tensor]],
        mask_ratio: torch.Tensor,
    ):
        loss = torch.tensor(0.0, device=real[0][0].device, dtype=real[0][0].dtype)
        
        # L_D = E_x [1/K ∑_k 1/T_k ∑_t max(0, 1 - D_{k,t}(x))] + 
        #       E_x [1/K ∑_k 1/T_k ∑_t max(0, 1 + D_{k,t}(G(x)))]
        
        K = len(real)  # The number of discriminators

        # Real data loss: max(0, 1 - D(x))
        real_loss_total = torch.tensor(0.0, device=real[0][0].device, dtype=real[0][0].dtype)
        for scale in real:
            real_loss_total += masked_mean_from_ratios(F.relu(1 - scale[-1]), mask_ratio)

        # Fake data loss: max(0, 1 + D(G(x)))
        fake_loss_total = torch.tensor(0.0, device=fake[0][0].device, dtype=fake[0][0].dtype)
        for scale in fake:
            fake_loss_total += masked_mean_from_ratios(F.relu(1 + scale[-1]), mask_ratio)
        
        loss = (real_loss_total + fake_loss_total) / K
        return loss


class GeneratorLoss(nn.Module):
    def forward(self, fake: list[list[torch.Tensor]], mask_ratio: torch.Tensor):
        loss = torch.tensor(0.0, device=fake[0][0].device, dtype=fake[0][0].dtype)
        
        # L_G^adv = E_x [1/K ∑_{k,t} 1/T_k max(0, 1 - D_{k,t}(G(x)))]
        K = len(fake)  # The number of discriminators
        total_loss = torch.tensor(0.0, device=fake[0][0].device, dtype=fake[0][0].dtype)
        
        fake_losses = []
        for scale in fake:
            fake_loss = masked_mean_from_ratios(F.relu(1 - scale[-1]), mask_ratio)
            fake_losses.append(fake_loss.item())
            total_loss += fake_loss

        # Normalize by K
        loss = total_loss / K
            
        return loss


class FeatureLoss(nn.Module):
    def __init__(self, n_blocks=3, n_features=16, n_layers=4):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_features = n_features
        self.n_layers = n_layers

    def forward(
        self,
        real: list[list[torch.Tensor]],
        fake: list[list[torch.Tensor]],
        mask_ratio: torch.Tensor,
    ):
        loss = torch.tensor(0.0, device=real[0][0].device, dtype=real[0][0].dtype)
        
        for i in range(self.n_blocks):
            for j in range(len(fake[i]) - 1):
                loss += masked_mean_from_ratios(
                torch.abs(fake[i][j] - real[i][j].detach()), mask_ratio
            )
        num_layers = len(fake[0]) - 1
        loss = loss / (self.n_blocks * num_layers)
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        mel_bins=64,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins
        self.gradient_checkpointing = gradient_checkpointing
        self.epsilon = 1e-6

    def _calculate_for_scale(self):
        def custom_run(*inputs):
            original, generated, s_exp, mask_ratio = (
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
            )
            s = 2**s_exp
            n_fft = 2**11
            window_size = s
            hop_length = int(s / 4)
            mel_spectrogram = MelSpectrogram(
                sample_rate=self.sample_rate,
                win_length=window_size,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.mel_bins,
            ).to(original.device)
            orig_audio_spec = mel_spectrogram(original)
            generated_audio_spec = mel_spectrogram(generated)
            
            l1_loss = torch.abs(orig_audio_spec - generated_audio_spec).sum()
            log_diff = torch.log(orig_audio_spec + self.epsilon) - torch.log(generated_audio_spec + self.epsilon)

            l2_log_loss = torch.sqrt(torch.sum(log_diff**2, dim=1)).sum()

            alpha_s = torch.sqrt(torch.tensor(s) / 2).to(original.device)
            return l1_loss + alpha_s * l2_log_loss

        return custom_run

    def forward(
        self, original: torch.Tensor, generated: torch.Tensor, mask_ratio: torch.Tensor
    ):
        assert original.shape == generated.shape
        loss = torch.tensor(0.0, device=original.device, dtype=original.dtype)
        for s_exp in range(6, 12):
            if self.gradient_checkpointing:
                loss += checkpoint(
                    self._calculate_for_scale(),
                    original,
                    generated,
                    s_exp,
                    mask_ratio,
                    use_reentrant=False,
                )
            else:
                loss += self._calculate_for_scale()(
                    original, generated, s_exp, mask_ratio
                )
        return loss
