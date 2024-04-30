import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
import numpy as np


def estimate(
    signal: torch.Tensor,
    sample_rate: float,
    frame_length: int,
    frame_stride: int,
    pitch_max: float = 20000,
    thresholds: Iterable[float] = (0.05, 0.1, 0.15),
    whitening: bool = True
) -> torch.Tensor:
    """Estimates the pitch (fundamental frequency) of a signal.

    :param signal: the signal vector (1D) or [..., time] tensor to analyze.
    :param sample_rate: sample rate, in Hz, of the signal.
    :param frame_length: number of samples per frame.
    :param frame_stride: overlapping window stride (number of samples), which determines the number of pitch values
                         returned
    :param pitch_max: expected upper bound of the pitch
    :param thresholds: A list of harmonic threshold values (see paper)
    :param whitening: Whether to apply whitening on the estimated period per utterance.
    :return: PyTorch tensor of shape [..., number_of_frames, number_of_thresholds * 3] as for each utterance, for each
      frame in the utterance, and for each threshold, we compute 3 things:
      1. The estimated pitch or 0 for aperiodic frames (and if whitening is true, it is normalized).
      2. Unvoiced predicate (1 for frames with an estimated pitch and 0 for aperiodic frames.
      3. The cumulative mean normalized difference value at the estimated period.
    """

    signal = torch.as_tensor(signal)

    # convert frequencies to samples, ensure windows can fit 2 whole periods
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(frame_length / 2)
    assert tau_min < tau_max

    # compute the fundamental periods
    frames = _frame(signal, frame_length, frame_stride)
    cmdf = _diff(frames, tau_max)[..., tau_min:]

    outputs = []
    for threshold in thresholds:
        tau = _search(cmdf, tau_max, threshold)

        # compute f0 by converting the periods to frequencies (if periodic).
        f0_estimation = torch.where(
            tau > 0,
            sample_rate / (tau + tau_min + 1).type(signal.dtype),
            torch.tensor(0, device=tau.device).type(signal.dtype),
        )
        if whitening:
            # Normalize each row by subtracting mean and dividing by std
            mean = f0_estimation.mean(dim=-1, keepdim=True)
            std = f0_estimation.std(dim=-1, keepdim=True)
            std_safe = torch.where(std > 0, std, torch.tensor(1e-8).to(std.device))
            f0_estimation = (f0_estimation - mean) / std_safe

        outputs.append(f0_estimation)

        # Compute the unvoiced signal predicate.
        unvoiced_predicate = torch.where(
            tau > 0,
            torch.tensor(1, device=tau.device).type(signal.dtype),
            torch.tensor(0, device=tau.device).type(signal.dtype),
        )
        outputs.append(unvoiced_predicate)

        # Compute the cumulative mean normalized difference value at the estimated period.
        cmdf_at_tau = torch.gather(cmdf, -1, tau.unsqueeze(-1)).squeeze(-1)
        outputs.append(cmdf_at_tau)

    output = torch.stack(outputs, dim=-1)
    return output


def _frame(signal: torch.Tensor, frame_length: int, frame_stride: int) -> torch.Tensor:
    # window the signal into overlapping frames, padding to at least 1 frame
    if signal.shape[-1] < frame_length:
        signal = torch.nn.functional.pad(signal, [0, frame_length - signal.shape[-1]])
    return signal.unfold(dimension=-1, size=frame_length, step=frame_stride)


def _diff(frames: torch.Tensor, tau_max: int) -> torch.Tensor:
    # compute the frame-wise autocorrelation using the FFT
    fft_size = 2 ** (-int(-np.log(frames.shape[-1]) // np.log(2)) + 1)
    fft = torch.fft.rfft(frames, fft_size, dim=-1)
    corr = torch.fft.irfft(fft * fft.conj())[..., :tau_max]

    # difference function (equation 6)
    sqrcs = torch.nn.functional.pad((frames * frames).cumsum(-1), [1, 0])
    corr_0 = sqrcs[..., -1:]
    corr_tau = sqrcs.flip(-1)[..., :tau_max] - sqrcs[..., :tau_max]
    diff = corr_0 + corr_tau - 2 * corr

    # cumulative mean normalized difference function (equation 8)
    return (
        diff[..., 1:]
        * torch.arange(1, diff.shape[-1], device=diff.device)
        / torch.maximum(
            diff[..., 1:].cumsum(-1),
            torch.tensor(1e-5, device=diff.device),
        )
    )


def _search(cmdf: torch.Tensor, tau_max: int, threshold: float) -> torch.Tensor:
    # mask all periods after the first cmdf below the threshold
    # if none are below threshold (argmax=0), this is a non-periodic frame
    first_below = (cmdf < threshold).int().argmax(-1, keepdim=True)
    first_below = torch.where(first_below > 0, first_below, tau_max)
    beyond_threshold = torch.arange(cmdf.shape[-1], device=cmdf.device) >= first_below

    # mask all periods with upward sloping cmdf to find the local minimum
    increasing_slope = torch.nn.functional.pad(cmdf.diff() >= 0.0, [0, 1], value=1)

    # find the first period satisfying both constraints
    return (beyond_threshold & increasing_slope).int().argmax(-1)


class F0Estimator(nn.Module):
    def __init__(self, sample_rate: int = 16_000, frame_length_ms: int = 20,
                 yin_thresholds: Iterable[float] = (0.05, 0.1, 0.15), whitening: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = int(self.sample_rate // (1 / frame_length_ms * 1000))
        self.yin_thresholds = yin_thresholds
        self.whitening = whitening

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.samples_per_frame, self.samples_per_frame), "constant", 0)
        f0 = estimate(x, self.sample_rate, self.samples_per_frame * 3, self.samples_per_frame,
                      thresholds=self.yin_thresholds, whitening=self.whitening,
                      )
        return f0


if __name__ == '__main__':
    x = torch.rand(4, 320003)
    print(f"{x.shape=}")
    F0 = F0Estimator(16000, 20)
    F0.forward(x)
